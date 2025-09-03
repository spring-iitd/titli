import torch
import pickle
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, to_tree

import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, confusion_matrix, 
                             accuracy_score, roc_curve, auc)

from titli.ids.base_ids import PyTorchModel
from titli.utils.data import create_directories


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class Autoencoder(nn.Module):
    def __init__(self, W, hbias, vbias):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(W.shape[0], W.shape[1], bias=True)
        self.encoder.weight.data = torch.from_numpy(W.T).float()
        self.encoder.bias.data = torch.from_numpy(hbias).float()
        self.decoder = nn.Linear(W.shape[1], W.shape[0], bias=True)
        self.decoder.weight.data = torch.from_numpy(W).float()
        self.decoder.bias.data = torch.from_numpy(vbias).float()
        self.rmse = RMSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class _TorchKitNET(nn.Module):
    def __init__(self, tail_weights, head_weight, clusters, num_features):
        super(_TorchKitNET, self).__init__()

        self.clusters = clusters
        # add the tails of the autoencoders
        self.tails = nn.ModuleList([Autoencoder(weight['W'], weight['hbias'], weight['vbias']) for weight in tail_weights])

        # add the head of the autoencoder
        self.head = Autoencoder(head_weight['W'], head_weight['hbias'], head_weight['vbias'])

        self.num_features = num_features
        self.rmse = RMSELoss()  # Add RMSE loss function

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_features)

        x_clusters = [
            torch.index_select(x, 1, torch.tensor(c, dtype=torch.long, device=x.device)) for c in self.clusters
        ]

        tail_losses = []
        for tail, c in zip(self.tails, x_clusters):
            output = tail(c)
            # Calculate per-sample reconstruction error
            loss = torch.sqrt(torch.mean((output - c) ** 2, dim=1))  # Per sample error
            tail_losses.append(loss)
        
        # Stack to get [batch_size, num_autoencoders]
        tails = torch.stack(tail_losses, dim=1)
        
        # Pass through head autoencoder
        head_output = self.head(tails)
        
        # Calculate reconstruction error from head autoencoder (this is the final anomaly score)
        head_rmse = torch.sqrt(torch.mean((head_output - tails) ** 2, dim=1))
        
        # Return the final anomaly scores and intermediate values for debugging
        return head_rmse, tails


class BaseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.sigmoid(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x


class TorchKitNET(nn.Module):
    def __init__(self, clusters: list, norms_path: str):
        super(TorchKitNET, self).__init__()
        self.dataset = "PcapDatasetRaw"
        self.input_dim = sum([len(c) for c in clusters])
        self.hr = 0.75
        self.clusters = clusters
        self.rmse = RMSELoss()
        self.tails = nn.ModuleList([BaseAutoencoder(len(c), int(np.ceil(len(c) * self.hr))) for c in clusters])
        self.head = BaseAutoencoder(len(clusters), int(np.ceil(len(clusters) * self.hr)))
        
        # Load normalization parameters
        if os.path.exists(norms_path):
            with open(norms_path, "rb") as f:
                self.norm_params = pickle.load(f)
        else:
            # Create dummy norm params if file doesn't exist
            self.norm_params = {}
            for c in clusters:
                self.norm_params[f"norm_max_{c[0]}"] = 1.0
                self.norm_params[f"norm_min_{c[0]}"] = 0.0
            self.norm_params["norm_max_output"] = 1.0
            self.norm_params["norm_min_output"] = 0.0

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x_clusters = []
        for c in self.clusters:
            norm_max = torch.tensor(self.norm_params[f"norm_max_{c[0]}"], dtype=torch.float32).to(x.device)
            norm_min = torch.tensor(self.norm_params[f"norm_min_{c[0]}"], dtype=torch.float32).to(x.device)

            x_cluster = torch.index_select(x, 1, torch.tensor(c, dtype=torch.long).to(x.device))
            x_cluster = (x_cluster - norm_min) / (norm_max - norm_min + 0.0000000000000001)
            x_cluster = x_cluster.float()

            x_clusters.append(x_cluster)

        tail_losses = []
        for tail, c in zip(self.tails, x_clusters):
            output = tail(c)
            loss = self.rmse(output, c)
            if loss.data == 0:
                loss.data = torch.tensor(1e-2, dtype=torch.float32).to(loss.device)
            tail_losses.append(loss)

        tails = torch.stack(tail_losses)

        # normalize the tails
        norm_max = torch.tensor(self.norm_params["norm_max_output"], dtype=torch.float32).to(x.device)
        norm_min = torch.tensor(self.norm_params["norm_min_output"], dtype=torch.float32).to(x.device)
        tails = (tails - norm_min) / (norm_max - norm_min + 0.0000000000000001)
        tails = tails.float()
        head_output = self.head(tails)

        return head_output, tails


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def squeeze_features(fv, precision):
    """rounds features to siginificant figures"""
    return np.around(fv, decimals=precision)


def quantize(x, k):
    n = 2**k - 1
    return np.round(np.multiply(n, x))/n


def quantize_weights(w, k):
    x = np.tanh(w)
    q = x / np.max(np.abs(x)) * 0.5 + 0.5
    return 2 * quantize(q, k) - 1


class dA_params:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0,
                 gracePeriod=10000, hiddenRatio=None, normalize=True,
                 input_precision=None, quantize=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio
        self.normalize = normalize
        self.quantize = quantize
        self.input_precision = input_precision
        if quantize:
            self.q_wbit, self.q_abit = quantize


class dA:
    def __init__(self, params):
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(np.ceil(
                self.params.n_visible * self.params.hiddenRatio))

        # for 0-1 normlaization
        self.norm_max = np.ones((self.params.n_visible,)) * -np.inf
        self.norm_min = np.ones((self.params.n_visible,)) * np.inf
        self.n = 0

        self.rng = np.random.RandomState(1234)

        a = 1. / self.params.n_visible
        self.W = np.array(self.rng.uniform(
            low=-a, high=a, size=(self.params.n_visible, self.params.n_hidden)))

        if self.params.quantize:
            self.W = quantize_weights(self.W, self.params.q_wbit)

        self.hbias = np.zeros(self.params.n_hidden)
        self.vbias = np.zeros(self.params.n_visible)

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1
        return self.rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    def get_hidden_values(self, input):
        return sigmoid(np.dot(input, self.W) + self.hbias)

    def get_reconstructed_input(self, hidden):
        return sigmoid(np.dot(hidden, self.W.T) + self.vbias)

    def train(self, x):
        self.n = self.n + 1

        if self.params.normalize:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

        if self.params.input_precision:
            x = squeeze_features(x, self.params.input_precision)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x

        y = self.get_hidden_values(tilde_x)
        if self.params.quantize:
            y = quantize(y, self.params.q_abit)

        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = np.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = np.outer(tilde_x.T, L_h1) + np.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * L_hbias
        self.vbias += self.params.lr * L_vbias

        if self.params.quantize:
            self.W = quantize_weights(self.W, self.params.q_wbit)
            self.hbias = quantize_weights(self.hbias, self.params.q_wbit)
            self.vbias = quantize_weights(self.vbias, self.params.q_wbit)

        return np.sqrt(np.mean(L_h2**2))

    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        try:
            if self.params.quantize:
                y = quantize(y, self.params.q_abit)
        except AttributeError:
            pass
        z = self.get_reconstructed_input(y)
        return z

    def get_params(self):
        return {"W": self.W, "hbias": self.hbias, "vbias": self.vbias}

    def set_params(self, new_param):
        self.W = new_param["W"]
        self.hbias = new_param["hbias"]
        self.vbias = new_param["vbias"]

    def execute(self, x):
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            try:
                if self.params.normalize:
                    x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
                if self.params.input_precision:
                    x = squeeze_features(x, self.params.input_precision)
            except AttributeError:
                pass

            z = self.reconstruct(x)
            rmse = np.sqrt(((x - z) ** 2).mean())
            return rmse

    def inGrace(self):
        return self.n < self.params.gracePeriod


class corClust:
    """A helper class for NumpyKitNET which performs a correlation-based incremental clustering of the dimensions in X"""
    def __init__(self, n):
        self.n = n
        self.c = np.zeros(n)
        self.c_r = np.zeros(n)
        self.c_rs = np.zeros(n)
        self.C = np.zeros((n, n))
        self.N = 0

    def update(self, x):
        self.N += 1
        self.c += x
        c_rt = x - self.c/self.N
        self.c_r += c_rt
        self.c_rs += c_rt**2
        self.C += np.outer(c_rt, c_rt)

    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt==0] = 1e-100
        D = 1 - self.C/C_rs_sqrt
        D[D<0] = 0
        return D

    def cluster(self, maxClust):
        D = self.corrDist()
        Z = linkage(D[np.triu_indices(self.n, 1)])
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        map = self.__breakClust__(to_tree(Z), maxClust)
        return map

    def __breakClust__(self, dendro, maxClust):
        if dendro.count <= maxClust:
            return [dendro.pre_order()]
        return self.__breakClust__(dendro.get_left(), maxClust) + self.__breakClust__(dendro.get_right(), maxClust)


class NumpyKitNET:
    """NumpyKitNET - A lightweight online anomaly detector based on an ensemble of autoencoders"""
    def __init__(self, n, max_autoencoder_size=10, FM_grace_period=None, AD_grace_period=10000,
                 learning_rate=0.1, hidden_ratio=0.75, feature_map=None, normalize=True,
                 input_precision=None, quantize=None, model_path="kitnet.pkl"):

        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        self.input_precision = input_precision
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n
        self.normalize = normalize
        self.n_trained = 0
        self.n_executed = 0
        self.v = feature_map
        self.ensembleLayer = []
        self.outputLayer = None
        self.quantize = quantize
        self.model_path = model_path
        self.norm_params_path = model_path.replace(".pkl", "_norm_params.pkl")
        
        if self.v is None:
            print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        
        self.FM = corClust(self.n)

    def process(self, x):
        if x.all() == -1:
            return 0.

        if self.n_trained > self.FM_grace_period + self.AD_grace_period:
            return self.execute(x)
        else:
            self.train(x)
            return 0.0

    def train(self, x):
        if self.n_trained <= self.FM_grace_period and self.v is None:
            self.FM.update(x)
            if self.n_trained == self.FM_grace_period:
                self.v = self.FM.cluster(self.m)
                self.__createAD__()
                print("The Feature-Mapper found a mapping: "+str(self.n)+ \
                      " features to "+str(len(self.v))+" autoencoders.")
                print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        else:
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].train(xi)
            rmse = self.outputLayer.train(S_l1)

            norm_params = {}
            for a in range(len(self.ensembleLayer)):
                norm_params[f"norm_min_{self.v[a][0]}"] = self.ensembleLayer[a].norm_min
                norm_params[f"norm_max_{self.v[a][0]}"] = self.ensembleLayer[a].norm_max
            
            norm_params["norm_min_output"] = self.outputLayer.norm_min
            norm_params["norm_max_output"] = self.outputLayer.norm_max

            with open(self.norm_params_path, 'wb') as f:
                pickle.dump(norm_params, f)

            if self.n_trained == self.AD_grace_period + self.FM_grace_period:
                print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
        self.n_trained += 1

    def execute(self, x):
        if self.v is None:
            raise RuntimeError('NumpyKitNET Cannot execute x, because a feature mapping has not yet been learned')
        else:
            self.n_executed += 1
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].execute(xi)
            return self.outputLayer.execute(S_l1)

    def __createAD__(self):
        for map in self.v:
            params = dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0,
                              hiddenRatio=self.hr, normalize=self.normalize,
                              input_precision=self.input_precision, quantize=self.quantize)
            self.ensembleLayer.append(dA(params))

        params = dA_params(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr,
                          normalize=self.normalize, quantize=self.quantize, input_precision=self.input_precision)
        self.outputLayer = dA(params)

    def get_params(self):
        return_dict = {"ensemble": []}
        for i in range(len(self.ensembleLayer)):
            return_dict["ensemble"].append(self.ensembleLayer[i].get_params())
        return_dict["output"] = self.outputLayer.get_params()
        return return_dict

    def set_params(self, new_param):
        for i in range(len(new_param["ensemble"])):
            self.ensembleLayer[i].set_params(new_param["ensemble"][i])
        self.outputLayer.set_params(new_param["output"])


class KitNET(PyTorchModel):
    """
    Standardized KitNET model that follows the PyTorchModel interface.
    
    This model uses a two-phase approach:
    1. Training phase: Uses the original numpy-based NumpyKitNET implementation
    2. Inference phase: Uses the PyTorch implementation for faster inference
    """
    
    def __init__(self, dataset_name, input_size, device, titles="default", 
                 max_autoencoder_size=10, FM_grace_ratio=0.2, AD_grace_ratio=0.8,
                 learning_rate=0.1, hidden_ratio=0.75):
        super().__init__(dataset_name, input_size, device)
        self.title = titles
        self.max_autoencoder_size = max_autoencoder_size
        self.FM_grace_ratio = FM_grace_ratio  # 0.2 = 20% of packets for feature mapping
        self.AD_grace_ratio = AD_grace_ratio  # 0.8 = 80% of packets for anomaly detection training
        self.learning_rate = learning_rate
        self.hidden_ratio = hidden_ratio
        
        # These will be calculated based on training data size
        self.FM_grace_period = None
        self.AD_grace_period = None
        
        # Paths for saving numpy and torch models
        self.numpy_model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pkl"
        self.norm_params_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}_norm_params.pkl"
        self.torch_model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pth"

        # Initialize the numpy NumpyKitNET model
        self.numpy_model = None
        self.torch_model = None
        self.clusters = None
        
        # Override epochs for KitNET (it uses grace periods instead)
        self.epochs = 1
    
    def __name__(self):
        return "KitNET"

    def get_params(self):
        """Get the model parameters"""
        return {
            "FM_grace_period": self.FM_grace_period,
            "AD_grace_period": self.AD_grace_period,
            "max_autoencoder_size": self.max_autoencoder_size,
            "learning_rate": self.learning_rate,
            "hidden_ratio": self.hidden_ratio
        }

    def get_model(self):
        """Return a placeholder - the actual model will be created during training"""
        return None
    
    def train_model(self, train_loader):
        """Train the KitNET model using the original numpy implementation"""
        print("Starting KitNET training with numpy implementation...")
        
        # Collect all training data
        all_train_data = []
        for inputs, _ in tqdm(train_loader, desc="Collecting training data"):
            all_train_data.append(inputs.numpy())
        all_train_data = np.concatenate(all_train_data, axis=0)
        
        # Calculate grace periods based on training data size
        num_packets = all_train_data.shape[0]
        self.FM_grace_period = int(np.floor(self.FM_grace_ratio * num_packets))
        self.AD_grace_period = int(np.floor(self.AD_grace_ratio * num_packets))
        
        print(f"Training data size: {num_packets} packets")
        print(f"FM grace period: {self.FM_grace_period} packets ({self.FM_grace_ratio*100}%)")
        print(f"AD grace period: {self.AD_grace_period} packets ({self.AD_grace_ratio*100}%)")
        
        # Fit scaler
        self.scaler.fit(all_train_data)
        print(f"Fitted scaler on {all_train_data.shape[0]} samples")
        
        # Initialize numpy NumpyKitNET
        self.numpy_model = NumpyKitNET(
            n=self.input_size,
            max_autoencoder_size=self.max_autoencoder_size,
            FM_grace_period=self.FM_grace_period,
            AD_grace_period=self.AD_grace_period,
            learning_rate=self.learning_rate,
            hidden_ratio=self.hidden_ratio,
            model_path=self.numpy_model_path
        )
        
        # Train the numpy model
        print("Training numpy NumpyKitNET model...")
        for i, sample in enumerate(tqdm(all_train_data, desc="Training NumpyKitNET")):
            # Apply scaling
            sample_scaled = self.scaler.transform(sample.reshape(1, -1)).flatten()
            self.numpy_model.process(sample_scaled)
            
        # Save the numpy model
        self._save_numpy_model()
        
        # Convert to PyTorch model
        self._convert_to_torch()
        
        # Calculate threshold using the torch model
        self.calculate_threshold(train_loader)

        self._calculate_threshold_pytorch(train_loader)
        
        print("KitNET training completed!")
    
    def _save_numpy_model(self):
        """Save the trained numpy model and its parameters"""
        os.makedirs(os.path.dirname(self.numpy_model_path), exist_ok=True)
        
        # Save the main model
        with open(self.numpy_model_path, 'wb') as f:
            pickle.dump({
                'model': self.numpy_model,
                'scaler': self.scaler,
                'clusters': self.numpy_model.v
            }, f)
        
        print(f"Numpy model saved to {self.numpy_model_path}")
    
    def _convert_to_torch(self):
        """Convert the trained numpy model to PyTorch"""
        if self.numpy_model is None or self.numpy_model.v is None:
            raise RuntimeError("Numpy model must be trained before conversion")
        
        # Store clusters for later use
        self.clusters = self.numpy_model.v
        
        # Get weights from numpy model
        weights = self.numpy_model.get_params()
        
        # Create _TorchKitNET model (backup model using weights)
        self.torch_model_backup = _TorchKitNET(
            weights["ensemble"], 
            weights["output"], 
            self.clusters, 
            self.input_size
        )
        self.torch_model_backup.to(self.device)
        
        # Create TorchKitNET model (main model using normalization parameters)
        self.torch_model = TorchKitNET(self.clusters, self.norm_params_path)
        self.torch_model.to(self.device)
        
        print("Successfully converted to PyTorch model")
    
    def forward(self, x):
        """Forward pass using the PyTorch model"""
        if self.torch_model is None:
            raise RuntimeError("PyTorch model not initialized. Train or load model first.")
        return self.torch_model(x)
    
    def calculate_threshold(self, train_loader):
        """Calculate threshold using the numpy model for consistency"""
        if self.numpy_model is None:
            raise RuntimeError("Numpy model not initialized")
            
        print("Calculating threshold using numpy model...")
        reconstruction_errors = []
        
        # Use numpy model for threshold calculation to ensure consistency
        for inputs, _ in tqdm(train_loader, desc="Calculating threshold"):
            for sample in inputs.numpy():
                sample_scaled = self.scaler.transform(sample.reshape(1, -1)).flatten()
                # Use numpy model's execute method
                rmse = self.numpy_model.execute(sample_scaled)
                if rmse is not None:  # Only add if not in training phase
                    reconstruction_errors.append(rmse)
        
        if len(reconstruction_errors) == 0:
            # Fallback: use PyTorch model
            print("No samples from numpy model (still training), using PyTorch model...")
            return self._calculate_threshold_pytorch(train_loader)
        
        # Set threshold at 95th percentile
        self.threshold = np.percentile(reconstruction_errors, 95)
        print(f"Threshold set to: {self.threshold}")
    
    def _calculate_threshold_pytorch(self, train_loader):
        """Fallback threshold calculation using PyTorch TorchKitNET model"""
        self.torch_model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Calculating threshold (PyTorch)"):
                # Scale inputs before feeding to TorchKitNET
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs.cpu().numpy())
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device)
                
                # Get outputs from TorchKitNET
                head_output, tails = self.torch_model(inputs_scaled)
                
                # Calculate RMSE like in the original test_ids.py
                criterion = RMSELoss()
                loss = criterion(head_output, tails)
                
                if loss.data == 0:
                    loss.data = torch.tensor(1e-2, dtype=torch.float32).to(loss.device)
                
                reconstruction_errors.append(loss.item())
        
        # Set threshold at 95th percentile
        self.threshold = np.percentile(reconstruction_errors, 95)
        print(f"Threshold set to: {self.threshold}")
    
    def infer(self, test_loader):
        """Inference using the PyTorch model"""
        if self.torch_model is None:
            raise RuntimeError("PyTorch model not initialized. Train or load model first.")
        
        print(f"Using threshold: {self.threshold}")
        self.torch_model.eval()
        reconstruction_errors = []
        y_test = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Inference"):
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs.cpu().numpy())
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device)
                
                # Process each sample in the batch individually
                for i in range(inputs_scaled.shape[0]):
                    single_input = inputs_scaled[i:i+1]  # Keep batch dimension
                    head_output, tail_losses = self.torch_model(single_input)
                    
                    # Calculate RMSE like in the threshold calculation
                    criterion = RMSELoss()
                    loss = criterion(head_output, tail_losses)
                    
                    if loss.data == 0:
                        loss.data = torch.tensor(1e-2, dtype=torch.float32).to(loss.device)
                    
                    reconstruction_errors.append(loss.item())
                    y_test.append(labels[i].item())
                    y_pred.append(int(loss.item() > self.threshold))
        
        return y_test, y_pred, reconstruction_errors
    
    def predict_single(self, features):
        """Predict anomaly for a single sample"""
        if self.torch_model is None:
            raise RuntimeError("PyTorch model not initialized")
            
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif isinstance(features, torch.Tensor) and features.dim() == 1:
            features = features.unsqueeze(0)
            
        features = features.to(self.device)
        
        # Scale features
        features_scaled = self.scaler.transform(features.cpu().numpy())
        features_scaled = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        
        self.torch_model.eval()
        with torch.no_grad():
            head_outputs, tail_losses = self.torch_model(features_scaled)
            # Both are now [1, num_autoencoders]
            # Calculate RMSE between head output and tail losses
            head_rmse = torch.sqrt(torch.mean((head_outputs - tail_losses) ** 2, dim=1))
            error = head_rmse.item()
            is_anomaly = (error > self.threshold)
            
        return is_anomaly, error
    
    def save(self, model_path=None):
        """Save the complete model (scaler, threshold, torch model, clusters)"""
        if model_path is None:
            model_path = self.torch_model_path
        
        if self.torch_model is None:
            raise RuntimeError("No trained model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Also save the numpy model weights for reconstruction
        weights = None
        if self.numpy_model is not None:
            weights = self.numpy_model.get_params()
        
        torch.save({
            'model_state_dict': self.torch_model.state_dict(),
            'threshold': self.threshold,
            'scaler': self.scaler,
            'clusters': self.clusters,
            'input_size': self.input_size,
            'weights': weights,  # Save the numpy weights for reconstruction
            'model_params': {
                'max_autoencoder_size': self.max_autoencoder_size,
                'FM_grace_period': self.FM_grace_period,
                'AD_grace_period': self.AD_grace_period,
                'learning_rate': self.learning_rate,
                'hidden_ratio': self.hidden_ratio
            }
        }, model_path)
        
        print(f"KitNET model saved to {model_path}")
    
    def load(self, model_path=None):
        """Load the complete model"""
        if model_path is None:
            model_path = self.torch_model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Restore parameters
        self.threshold = checkpoint['threshold']
        self.scaler = checkpoint['scaler']
        self.clusters = checkpoint['clusters']
        weights = checkpoint.get('weights', None)
        
        # Restore model parameters
        model_params = checkpoint.get('model_params', {})
        self.max_autoencoder_size = model_params.get('max_autoencoder_size', 10)
        self.FM_grace_period = model_params.get('FM_grace_period', None)
        self.AD_grace_period = model_params.get('AD_grace_period', 10000)
        self.learning_rate = model_params.get('learning_rate', 0.1)
        self.hidden_ratio = model_params.get('hidden_ratio', 0.75)
        
        # Recreate PyTorch model
        if weights is not None:
            self.torch_model = _TorchKitNET(
                weights["ensemble"], 
                weights["output"], 
                self.clusters, 
                self.input_size
            )
        else:
            # Fallback: create with empty weights and load state dict
            self.torch_model = _TorchKitNET([], {}, self.clusters, self.input_size)
        
        self.torch_model.load_state_dict(checkpoint['model_state_dict'])
        self.torch_model.to(self.device)
        
        print(f"KitNET model loaded from {model_path}")
        return checkpoint
    
    def get_model_info(self):
        """Get information about the model structure"""
        if self.clusters is None:
            return "Model not trained yet"
        
        info = {
            'num_clusters': len(self.clusters),
            'cluster_sizes': [len(cluster) for cluster in self.clusters],
            'total_features': self.input_size,
            'max_autoencoder_size': self.max_autoencoder_size,
            'threshold': self.threshold
        }
        return info

    def evaluate(self, test_loader):
        """
        Evaluates the model on test data from a DataLoader using the numpy model for consistency.
        """
        if self.numpy_model is None:
            raise RuntimeError("Numpy model not initialized")
            
        print("Running evaluation using numpy model...")
        y_test = []
        y_pred = []
        reconstruction_errors = []
        
        sample_count = 0
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx+1}")
            
            # Process each sample individually with numpy model
            for i in range(inputs.shape[0]):
                sample = inputs[i].numpy()
                label = labels[i].numpy()
                
                # Scale input
                sample_scaled = self.scaler.transform(sample.reshape(1, -1)).flatten()
                
                # Get anomaly score using numpy model
                anomaly_score = self.numpy_model.execute(sample_scaled)
                
                if anomaly_score is not None:  # Only process if not in training phase
                    # Get prediction based on threshold
                    prediction = 1.0 if anomaly_score > self.threshold else 0.0
                    
                    # Store results
                    y_test.append(label)
                    y_pred.append(prediction)
                    reconstruction_errors.append(anomaly_score)
                    
                    sample_count += 1
                    if sample_count >= 10000:  # Limit to avoid too much processing
                        break
            
            if sample_count >= 10000:
                break
        
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Ensure labels are 1D
        if y_test.ndim > 1:
            y_test = y_test.ravel()
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        
        # Call the parent evaluate method
        return super().evaluate(y_test, y_pred, reconstruction_errors)


# Legacy compatibility - alias for the standardized version
class KitsuneIDS(KitNET):
    """Legacy alias for KitNET"""
    pass
