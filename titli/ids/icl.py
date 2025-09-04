from .base_ids import PyTorchModel


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import argparse
import pickle

from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, confusion_matrix, 
                             accuracy_score, roc_curve, auc)

import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

class ICL(PyTorchModel):
    def __init__(self, dataset_name, input_size, device, titles):
        self.title = titles
        self.n_features = 100
        self.kernel_size = 10
        self.rep_dim = 32
        self.tau = 0.01
        self.max_negatives = 1000
        # self.hidden_dims='16,4'

        super().__init__(dataset_name, input_size, device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(self.device)

    def forward(self, x):
        positives, query = self.positive_matrix_builder(x)
        positives = F.normalize(self.enc_g_net(positives), dim=-1)
        query = F.normalize(self.enc_f_net(query), dim=-1).unsqueeze(1)
        logit = self.cal_logit(query, positives)
        logit = logit.permute(0, 2, 1)
        correct_class = torch.zeros((logit.shape[0], logit.shape[2]), dtype=torch.long, device=x.device)
        loss = self.criterion(logit, correct_class).mean(dim=1)
        return loss
    
    def get_model(self):
        hidden_dims = '16,4'
        hidden_dims = [int(a) for a in hidden_dims.split(',')]
        self.enc_f_net = nn.Linear(self.n_features - self.kernel_size + 1, self.rep_dim)
        self.enc_g_net = nn.Linear(self.kernel_size, self.rep_dim)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def cal_logit(self, query, pos):
        batch_size, n_pos, _ = pos.shape  
        negative_index = torch.randperm(n_pos)[:min(self.max_negatives, n_pos)]
        negative = pos.permute(0, 2, 1)[:, :, negative_index]  
        pos_multiplication = (query * pos).sum(dim=2, keepdim=True)  
        neg_multiplication = torch.matmul(query, negative)  
        identity_matrix = torch.eye(n_pos, device=query.device).unsqueeze(0).repeat(batch_size, 1, 1)[:, :, negative_index]  
        neg_multiplication = neg_multiplication.masked_fill(identity_matrix == 1, -float('inf'))  
        logit = torch.cat((pos_multiplication, neg_multiplication), dim=2) / self.tau  
        return logit

    def positive_matrix_builder(self, x):
        idx = np.arange(self.n_features - self.kernel_size + 1)
        sub_idx = idx[:, None] + np.arange(self.kernel_size)
        matrix = x[:, sub_idx]  
        complement_matrix = x[:, idx]  
        return matrix.float(), complement_matrix.float()

    def calculate_threshold(self, val_loader):
        self.eval()
        scores = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = self.scaler.transform(x.cpu().numpy())  # Scale the data
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                scores.extend(self.forward(x).cpu().numpy().tolist())
        self.threshold = np.percentile(scores, 95)
    
        print(f"the threshold is :{self.threshold}")
        return self.threshold

    def train_model(self, train_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        all_train_data = []  # Collect all training data in a list
        for inputs, _ in train_loader:
            all_train_data.append(inputs.numpy())  # Convert tensor to numpy
        all_train_data = np.concatenate(all_train_data, axis=0)
        self.scaler.fit(all_train_data)
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0
            for x, _ in train_loader:
                x = x.to(self.device)
                x = self.scaler.transform(x.cpu().numpy())  # Scale the data
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                optimizer.zero_grad()
                loss = self.forward(x).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            # Calculate average loss using batch count instead of len(train_loader)
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        threshold = self.calculate_threshold(train_loader)
        print(f"Threshold calculated and saved: {threshold}")

    def infer(self, loader):
        if not self.threshold:
            print("Threshold not set. Please load or train before inferring.")
            return None

        print("Using the threshold of {:.2f}".format(self.threshold))
        self.eval()
        reconstruction_errors = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                x = self.scaler.transform(x.cpu().numpy())
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                loss = self.forward(x)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend((loss.cpu().numpy() > self.threshold).astype(int).tolist())
                reconstruction_errors.extend(loss.cpu().numpy().tolist())
        
        return y_true, y_pred, reconstruction_errors


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ICL model")
    parser.add_argument("--data-path", type=str, default="utils/weekday_20k.csv", help="Path to the dataset")
    parser.add_argument("--model-path", type=str, default="icl_model.pth", help="Path to save the trained model")
    parser.add_argument("--threshold-path", type=str, default="threshold.pkl", help="Path to save the calculated threshold")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(args.data_path)
    X, y = data.iloc[:, :-1].values.astype(np.float32), data.iloc[:, -1].values.astype(np.float32)
    tensor_data = TensorDataset(torch.tensor(X), torch.tensor(y))
    train_loader = DataLoader(tensor_data, batch_size=args.batch_size, shuffle=True)

    model = ICL()
    model.train_model(train_loader, train_loader, args.model_path)
    model.save_model(args.model_path)
    model.load_model(args.model_path)
    y_true, y_pred = model.infer(train_loader, device=device)
    print(len(y_true), len(y_pred))
    model.evaluate(y_true, y_pred)


if __name__ == "__main__":
    main()
