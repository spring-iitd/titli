from .base_ids import PyTorchModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm

from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, confusion_matrix, 
                             accuracy_score, roc_curve, auc)


import seaborn as sns

class SimpleScaler:
    def __init__(self):
        """
        Simple scaler: collect first N samples, fit once, use forever
        """
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X):
        """Fit scaler on collected data"""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        self.scaler.fit(X)
        self.is_fitted = True
        
    def transform(self, X):
        """Transform data using fitted scaler"""
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
            transformed = self.scaler.transform(X_np)
            return torch.tensor(transformed, dtype=torch.float32).to(X.device)
        else:
            return self.scaler.transform(X)
    
    def get_params(self):
        """Get scaler parameters"""
        if self.is_fitted:
            return {
                'mean': self.scaler.mean_,
                'scale': self.scaler.scale_,
                'is_fitted': self.is_fitted
            }
        return {'is_fitted': False}

class VAE(PyTorchModel):
    def __init__(self, dataset_name, input_size, device, titles):
        super().__init__(dataset_name, input_size, device)
        self.title = titles
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.simple_scaler = SimpleScaler()  # Simple scaler: collect, fit once, use always
        
        # Move model to the correct device
        self.to(self.device)
        
    def get_model(self):
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 8),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU()
        )
        self.fc_mu = nn.Linear(8, 2)
        self.fc_logvar = nn.Linear(8, 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, self.input_size),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        logvar = torch.clip(logvar, max=10.0)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Use same loss calculation as inference for consistency
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # Still mean for training stability
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld
    
    def reconstruction_error(self, recon_x, x):
        """Calculate per-sample reconstruction errors (used for threshold and inference)"""
        return F.mse_loss(recon_x, x, reduction='none').mean(dim=1)

    def train_model(self, train_loader):
        """
        Simple approach:
        1. Collect first N samples
        2. Fit scaler once
        3. Train model using fitted scaler
        """
        
        print("Phase 1: Collecting initial samples for scaler...")
        # Phase 1: Collect first N samples
        all_data = []
        max_samples = 5000
        collected_samples = 0
        
        for x, _ in tqdm(train_loader, desc="Collecting samples"):
            x = x.to(self.device)
            all_data.append(x.cpu().numpy())
            collected_samples += x.shape[0]
            
            if collected_samples >= max_samples:
                break
        
        # Concatenate and fit scaler
        all_data = np.vstack(all_data)
        print(f"Collected {all_data.shape[0]} samples for scaler fitting")
        self.simple_scaler.fit(all_data)
        print("Scaler fitted!")
        
        print("Phase 2: Training model with fitted scaler...")
        # Phase 2: Train model using fitted scaler
        for epoch in range(self.epochs):
            running_loss = 0.0
            batch_count = 0
            
            for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                x = x.to(self.device)
                
                # Transform using FITTED scaler (no more updates)
                x_scaled = self.simple_scaler.transform(x).to(self.device)
                
                self.optimizer.zero_grad()
                recon, mu, logvar = self(x_scaled)
                loss = self.loss_function(recon, x_scaled, mu, logvar)
                
                # Add gradient clipping to prevent NaN
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item()
                batch_count += 1
                
            # Calculate average loss using batch count instead of len(train_loader)
            avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
            
        print("Training completed!")
        self.calculate_threshold(train_loader)

    def calculate_threshold(self, loader):
        reconstruction_errors = []
        self.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                x_scaled = self.simple_scaler.transform(x).to(self.device)
                recon, _, _ = self(x_scaled)
                errors = self.reconstruction_error(recon, x_scaled)  # Use consistent method
                reconstruction_errors.extend(errors.cpu().numpy())
        self.threshold = np.percentile(reconstruction_errors, 95)

        print(f"Threshold: {self.threshold}")

    def infer(self, dataloader):
        self.eval()
        y_true, y_pred, reconstruction_errors = [], [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                x_scaled = self.simple_scaler.transform(x).to(self.device)
                recon, _, _ = self(x_scaled)
                errors = self.reconstruction_error(recon, x_scaled)  # Use consistent method
                errors_np = errors.cpu().numpy()
                reconstruction_errors.extend(errors_np)
                y_true.extend(y.numpy())
                y_pred.extend((errors_np > self.threshold).astype(int))

        return np.array(y_true), np.array(y_pred), reconstruction_errors

    def predict_single(self, features):
        """
        Predict anomaly for a single packet using fitted scaler
        Args:
            features: Single packet features (numpy array or tensor)
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif isinstance(features, torch.Tensor) and features.dim() == 1:
            features = features.unsqueeze(0)
            
        features = features.to(self.device)
        
        # Transform using fitted scaler (no updates)
        features_scaled = self.simple_scaler.transform(features).to(self.device)
        
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(features_scaled)
            error = F.mse_loss(recon, features_scaled, reduction='none').mean(dim=1)
            is_anomaly = (error.item() > self.threshold)
            
        return is_anomaly, error.item()

    def save(self, model_path=None):
        """Save model with scaler parameters"""
        if model_path is None:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}_{self.title}.pth"
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'threshold': self.threshold,
            'scaler_params': self.simple_scaler.get_params(),
            'input_size': self.input_size
        }, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path=None):
        """Load model with scaler parameters"""
        if model_path is None:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}_{self.title}.pth"
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        
        # Restore scaler parameters
        scaler_params = checkpoint['scaler_params']
        if scaler_params['is_fitted']:
            # Recreate the scaler with saved parameters
            from sklearn.preprocessing import StandardScaler
            self.simple_scaler.scaler = StandardScaler()
            self.simple_scaler.scaler.mean_ = scaler_params['mean']
            self.simple_scaler.scaler.scale_ = scaler_params['scale']
            self.simple_scaler.is_fitted = True
        
        print(f"Model loaded from {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate VAE model")
    parser.add_argument("--data-path", type=str, default="../../utils/weekday_20k.csv", help="Path to the dataset")
    parser.add_argument("--model-path", type=str, default="vae_model.pth", help="Path to save the trained model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training and evaluation")
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    input_size = X.shape[1]
   
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tensor_data = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(tensor_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Dataset 
    dataset_name = "uq-iot"
    
    model = VAE(dataset_name=dataset_name, input_size=100, device="cpu", titles="online")
    model.train_model(dataloader)
    model.save()
    model.load()
    y_true, y_pred, reconstruction_errors = model.infer(dataloader)
    print(len(y_true), len(y_pred))
    model.evaluate(y_true, y_pred, reconstruction_errors)
