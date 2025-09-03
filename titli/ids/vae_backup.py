
from .base_ids import PyTorchModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataL    def infer(self, dataloader):
        self.eval()
        y_true, y_pred, reconstruction_errors = [], [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                x_scaled = self.online_scaler.transform(x)  # Use online scaler
                recon, _, _ = self(x_scaled)
                loss = F.mse_loss(recon, x_scaled, reduction='none').mean(dim=1)
                errors = loss.cpu().numpy()
                reconstruction_errors.extend(errors)
                y_true.extend(y.numpy())
                y_pred.extend((errors > self.threshold).astype(int))

        return np.array(y_true), np.array(y_pred), reconstruction_errorsaset
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

class OnlineStandardScaler:
    def __init__(self, input_size):
        self.input_size = input_size
        self.n_samples = 0
        self.mean = np.zeros(input_size)
        self.var = np.zeros(input_size)
        self.std = np.ones(input_size)
        
    def partial_fit(self, X):
        """Update the scaler with a new batch of data using Welford's online algorithm"""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        batch_size = X.shape[0]
        
        if self.n_samples == 0:
            # First batch - initialize
            self.mean = np.mean(X, axis=0)
            self.var = np.var(X, axis=0)
            self.n_samples = batch_size
        else:
            # Update using Welford's algorithm for online variance calculation
            for x in X:
                self.n_samples += 1
                delta = x - self.mean
                self.mean += delta / self.n_samples
                delta2 = x - self.mean
                self.var += (delta * delta2 - self.var) / self.n_samples
        
        # Update standard deviation (avoid division by zero)
        self.std = np.sqrt(np.maximum(self.var, 1e-8))
        
    def transform(self, X):
        """Transform the data using current mean and std"""
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
            transformed = (X_np - self.mean) / self.std
            return torch.tensor(transformed, dtype=torch.float32).to(X.device)
        else:
            return (X - self.mean) / self.std
    
    def get_params(self):
        """Get current scaler parameters"""
        return {
            'mean': self.mean,
            'std': self.std,
            'var': self.var,
            'n_samples': self.n_samples
        }

class VAE(PyTorchModel):
    def __init__(self, dataset_name, input_size, device, titles):
        super().__init__(dataset_name, input_size, device)
        self.title = titles
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.online_scaler = OnlineStandardScaler(input_size)  # Use online scaler instead
        
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
        recon_loss = self.criterion(recon_x, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def train_model(self, train_loader):
        """Train with online scaling - scaler updates with each batch"""
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")
            running_loss = 0.0
            
            for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                x = x.to(self.device)
                
                # Update scaler with current batch (simulating live packets)
                self.online_scaler.partial_fit(x)
                
                # Transform using updated scaler
                x_scaled = self.online_scaler.transform(x)
                
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
                
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
            
        print(f"Scaler fitted on {self.online_scaler.n_samples} samples")
        self.calculate_threshold(train_loader)

    def calculate_threshold(self, loader):
        reconstruction_errors = []
        self.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                x_scaled = self.online_scaler.transform(x)  # Use online scaler
                recon, _, _ = self(x_scaled)
                loss = F.mse_loss(recon, x_scaled, reduction='none').mean(dim=1)
                reconstruction_errors.extend(loss.cpu().numpy())
        self.threshold = np.percentile(reconstruction_errors, 95)

        print(f"Threshold: {self.threshold}")
    

    def infer(self, dataloader):
        self.eval()
        y_true, y_pred, reconstruction_errors = [], [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                x = self.scaler.transform(x.cpu().numpy())  # Scale the data    
                x = torch.tensor(x, dtype=torch.float32).to(self.device)  # Convert back to tensor
                recon, _, _ = self(x)
                loss = F.mse_loss(recon, x, reduction='none').mean(dim=1)
                errors = loss.cpu().numpy()
                reconstruction_errors.extend(errors)
                y_true.extend(y.numpy())
                y_pred.extend((errors > self.threshold).astype(int))

        return np.array(y_true), np.array(y_pred), reconstruction_errors


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
    
    model = VAE(dataset_name=dataset_name, input_size=100, device="cpu")
    model.train_model(dataloader)
    model.save()
    model.load()
    y_true, y_pred= model.infer(dataloader)
    print(len(y_true), len(y_pred))
    model.evaluate(y_true, y_pred)
