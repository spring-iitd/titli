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
from base_ids import PyTorchModel
class VAE(PyTorchModel):
    def __init__(self):
        super().__init__("VAE",100,"cpu")
        self.input_size = 100
        self.device = "cpu"
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.threshold = 0.0
        self.model = "VAE"
        
    def get_model(self):
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, 32)
        self.fc_logvar = nn.Linear(64, 32)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_size),
            nn.Sigmoid()
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
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = self.criterion(recon_x, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def train_model(self, train_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        for epoch in range(1):
            running_loss = 0.0
            for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                x = x.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self(x)
                loss = self.loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader)}")
        self.calculate_threshold(train_loader)

    def calculate_threshold(self, loader):
        recon_errors = []
        self.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                recon, _, _ = self(x)
                loss = F.mse_loss(recon, x, reduction='none').mean(dim=1)
                recon_errors.extend(loss.cpu().numpy())
        self.threshold = np.percentile(recon_errors, 95)

        print(f"Threshold: {self.threshold}")
        threshold_file = "threshold_"+str(self.model_name)+".pkl"
        pickle.dump(self.threshold, open(threshold_file, 'wb')); print(f"Threshold saved to {threshold_file}")

    def infer(self, dataloader):
        self.eval()
        y_true, y_pred, recon_errors = [], [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                recon, _, _ = self(x)
                loss = F.mse_loss(recon, x, reduction='none').mean(dim=1)
                errors = loss.cpu().numpy()
                recon_errors.extend(errors)
                y_true.extend(y.numpy())
                y_pred.extend((errors > self.threshold).astype(int))
        reconstruction_errors = "reconstruction_error_"+str(self.model_name)+".pkl"
        pickle.dump(recon_errors, open(reconstruction_errors, 'wb')); print(f"Reconstruction errors saved to {reconstruction_errors}")
        return np.array(y_true), np.array(y_pred)

    
    def save(self, model_path):
        torch.save({
            "model_state_dict": self.state_dict(),
            "scaler": self.scaler,
            "threshold": self.threshold
        }, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint.get("scaler", StandardScaler())
        self.threshold = checkpoint.get("threshold", 0.0)
        self.eval()
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

    model = VAE()
    model.train_model(dataloader)
    model.save(args.model_path)
    model.load(args.model_path)
    y_true, y_pred= model.infer(dataloader)
    print(len(y_true), len(y_pred))
    model.evaluate(y_true, y_pred)
