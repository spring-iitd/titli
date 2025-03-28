from base_ids import PyTorchModel


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
    def __init__(self, n_features = 100, kernel_size = 10, hidden_dims='16,4', rep_dim=32, tau=0.01, max_negatives=1000):
        super(ICL, self).__init__("ICL",100,"cpu")
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.rep_dim = rep_dim
        self.tau = tau
        self.max_negatives = max_negatives

        hidden_dims = [int(a) for a in hidden_dims.split(',')]
        self.enc_f_net = nn.Linear(n_features - kernel_size + 1, rep_dim)
        self.enc_g_net = nn.Linear(kernel_size, rep_dim)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

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
        a = 10
    
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
                scores.extend(self.forward(x).cpu().numpy().tolist())
        threshold = np.percentile(scores, 95)
        threshold_path="threshold_"+str(self.model_name)+".pkl"
        with open(threshold_path, "wb") as f:
            pickle.dump(threshold, f)
        print(f"the threshold is :{threshold}")
        return threshold

    def train_model(self, train_loader, model_path="icl_model.pth", threshold_path="threshold.pkl", device="cpu", epochs=1, lr=1e-3):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0
            for x, _ in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                loss = self.forward(x).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        threshold = self.calculate_threshold(train_loader)
        print(f"Threshold calculated and saved: {threshold}")

    def infer(self, loader, device):
        # self.eval()
        y_true, y_pred, errors = [], [], []
        threshold_path="threshold_"+str(self.model_name)+".pkl"
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                loss = self.forward(x)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend((loss.cpu().numpy() > threshold).astype(int).tolist())
                errors.extend(loss.cpu().numpy().tolist())
        reconstruction_errors = "reconstruction_error_"+str(self.model_name)+".pkl"

        with open(reconstruction_errors, "wb") as f:
            pickle.dump(errors, f)
        return y_true, y_pred
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("the model is saved sucessfully")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print("the model is loaded successfully")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ICL model")
    parser.add_argument("--data-path", type=str, default="/home/kundan/titli/utils/weekday_20k.csv", help="Path to the dataset")
    parser.add_argument("--model-path", type=str, default="icl_model.pth", help="Path to save the trained model")
    parser.add_argument("--threshold-path", type=str, default="threshold.pkl", help="Path to save the calculated threshold")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    X, y = data.iloc[:, :-1].values.astype(np.float32), data.iloc[:, -1].values.astype(np.float32)
    tensor_data = TensorDataset(torch.tensor(X), torch.tensor(y))
    train_loader = DataLoader(tensor_data, batch_size=args.batch_size, shuffle=True)

    model = ICL()
    model.train_model(train_loader, train_loader,args.model_path)
    model.save_model(args.model_path)
    model.load_model(args.model_path)
    y_true, y_pred = model.infer(train_loader, device="cpu")
    print(len(y_true), len(y_pred))
    model.evaluate(y_true, y_pred)

if __name__ == "__main__":
    main()
