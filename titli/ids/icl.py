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

    def calculate_threshold(self, val_loader, threshold_path="threshold_icl.pkl"):
        self.eval()
        scores = []
        with torch.no_grad():
            for x, _ in val_loader:
                scores.extend(self.forward(x).cpu().numpy().tolist())
        threshold = np.percentile(scores, 95)
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
        threshold = self.calculate_threshold(train_loader, threshold_path)
        print(f"Threshold calculated and saved: {threshold}")

    def infer(self, loader, device, threshold_path="threshold.pkl"):
        # self.eval()
        y_true, y_pred, errors = [], [], []
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                loss = self.forward(x)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend((loss.cpu().numpy() > threshold).astype(int).tolist())
                errors.extend(loss.cpu().numpy().tolist())
        with open("error_icl.pkl", "wb") as f:
            pickle.dump(errors, f)
        return y_true, y_pred
    def evaluate(self,y_test, y_pred, device="cpu", cm_save_path="confusion_matrix_icl.png", roc_save_path="roc_curve.png"):
        """
        Evaluates the model on the test set, calculates evaluation metrics, and plots confusion matrix and ROC curve.
        """
        with open("threshold.pkl", 'rb') as f:
            threshold = pickle.load(f)
        with open("error_icl.pkl", 'rb') as g:
            reconstruction_errors = pickle.load(g)
        print("Using the threshold of {:.2f}".format(threshold))
    
        cm = confusion_matrix(y_test, y_pred)

        # Compute evaluation metrics
        f1 = round(f1_score(y_test, y_pred, zero_division=1), 3)
        precision = round(precision_score(y_test, y_pred, zero_division=1), 3)
        recall = round(recall_score(y_test, y_pred, zero_division=1), 3)
        accuracy = round(accuracy_score(y_test, y_pred), 3)

        # Print the evaluation metrics
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")

        # --- Confusion Matrix Plot ---
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt=",.0f", cmap="Blues", 
                        xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"], 
                        cbar=True)  # Add color bar explicitly

        # Format the annotations to use scientific notation
        for text in ax.texts:
            # Remove commas and convert to float, then format as scientific notation
            text_value = text.get_text().replace(',', '')  # Remove commas
            try:
                text.set_text(f'{float(text_value):.2e}')  # Format as scientific notation (e.g., 2.55E+05)
            except ValueError:
                continue  # Skip any annotation that can't be converted to float (e.g., if it's empty or NaN)

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(cm_save_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_save_path}")

        # --- ROC Curve and EER Calculation ---
        if np.sum(y_test) == 0 or np.sum(y_test) == len(y_test):
            print("Warning: ROC curve cannot be computed because y_test contains only one class.")
        else:
            fpr, tpr, thresholds = roc_curve(y_test, reconstruction_errors)
            roc_auc = auc(fpr, tpr)

            eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
            eer_threshold = thresholds[eer_index]
            eer = fpr[eer_index]

            # --- ROC Curve Plot ---
            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="blue")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.scatter(fpr[eer_index], tpr[eer_index], color='red', label=f"EER = {eer:.3f} at Threshold = {eer_threshold:.3f}")
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("ROC Curve with EER")
            plt.legend()
            plt.grid()
            plt.savefig(roc_save_path)
            plt.close()
            print(f"ROC curve saved to {roc_save_path}")

            # Display AUC and EER values in decimal format
            print(f"AUC: {roc_auc:.3f}, EER: {eer:.3f} at threshold {eer_threshold:.3f}")

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
