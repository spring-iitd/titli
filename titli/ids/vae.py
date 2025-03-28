import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import argparse

class ICL(nn.Module):
    def __init__(self, n_features, kernel_size, hidden_dims='16,4', rep_dim=32, tau=0.01, max_negatives=1000):
        super(ICL, self).__init__()
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
        query = F.normalize(self.enc_f_net(query), dim=-1)
        logit = self.cal_logit(query, positives)
        logit = logit.permute(0, 2, 1)
        correct_class = torch.zeros((logit.shape[0], logit.shape[2]), dtype=torch.long, device=x.device)
        loss = self.criterion(logit, correct_class).mean(dim=1)
        return loss

    def cal_logit(self, query, pos):
        batch_size, n_pos = query.shape[:2]
        negative_index = torch.randperm(n_pos)[:min(self.max_negatives, n_pos)]
        negative = pos.permute(0, 2, 1)[:, :, negative_index]
        pos_multiplication = (query * pos).sum(dim=2, keepdim=True)
        neg_multiplication = torch.matmul(query, negative)
        identity_matrix = torch.eye(n_pos, device=query.device).unsqueeze(0).repeat(batch_size, 1, 1)[:, :, negative_index]
        neg_multiplication.masked_fill_(identity_matrix == 1, -float('inf'))
        logit = torch.cat((pos_multiplication, neg_multiplication), dim=2) / self.tau
        return logit

    def positive_matrix_builder(self, x):
        idx = np.arange(self.n_features - self.kernel_size + 1)
        sub_idx = idx[:, None] + np.arange(self.kernel_size)
        comp_idx = np.array([np.setdiff1d(np.arange(self.n_features), row) for row in sub_idx])
        matrix = x[:, sub_idx]
        complement_matrix = x[:, comp_idx]
        return matrix, complement_matrix

    def train_model(self, train_loader, device, epochs, lr):
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

    def infer(self, loader, device):
        self.eval()
        y_true, y_pred, errors = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                loss = self.forward(x)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend((loss.cpu().numpy() > 0.5).astype(int).tolist())
                errors.extend(loss.cpu().numpy().tolist())
        return y_true, y_pred, errors

    def evaluate(self, test_loader, threshold, device):
        self.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                scores.extend(self.forward(x).cpu().numpy().tolist())
                labels.extend(y.cpu().numpy().tolist())
        predictions = np.array(scores) > threshold
        accuracy = np.mean(predictions == np.array(labels))
        return accuracy

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)

    def determine_threshold(self, val_loader, percentile=95, device='cpu'):
        self.eval()
        scores = []
        with torch.no_grad():
            for x, _ in val_loader:
                scores.extend(self.forward(x).cpu().numpy().tolist())
        return np.percentile(scores, percentile)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ICL model")
    parser.add_argument("--model-path", type=str, default="icl_model.pth", help="Path to save the trained model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training and evaluation")
    args = parser.parse_args()
    data_path = "../../utils/weekday_20k.csv"
    data = pd.read_csv(args.data_path)
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    input_size = X.shape[1]

    tensor_data = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(tensor_data, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = ICL(n_features=input_size, kernel_size=10).to(device)
    model.train_model(dataloader, device, args.epochs, args.lr)
    model.save_model(args.model_path)
    model.load_model(args.model_path, device)
    
    y_true, y_pred, errors = model.infer(dataloader, device)
    print(len(y_true), len(y_pred), len(errors))
    
    threshold = model.determine_threshold(dataloader, device=device)
    print(f"Determined Threshold: {threshold:.4f}")

    accuracy = model.evaluate(dataloader, threshold, device)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
