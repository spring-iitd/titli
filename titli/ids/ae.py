from base_ids import PyTorchModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class Autoencoder(PyTorchModel):
    input_size = 100
    device = "cpu"
    def __init__(self, input_size, device):
        super(Autoencoder, self).__init__("Autoencoder", input_size, device)

    def get_model(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Autoencoder model")
    # parser.add_argument("--data-path", type=str, default="/home/kundan/byte-me/data/cic_csv/final_output.csv", help="Path to the dataset")

    parser.add_argument("--data-path", type=str, default="/home/kundan/byte-me/data/cic_csv/cic-2023_chopped/Benign_Final/BenignTraffic.csv", help="Path to the dataset")
    parser.add_argument("--model-path", type=str, default="autoencoder.pth", help="Path to save the trained model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training and evaluation")
    args = parser.parse_args()
    
    model_path = args.model_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_size=100, device=device).to(device)

    data = pd.read_csv(args.data_path)
    feature, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
    
    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    dataset = TensorDataset(torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.train_model(dataloader)
    model.save(model_path)
    model.load(Autoencoder, model_path)
    
    model.evaluate(dataloader)

