import time
from matplotlib import ticker
import torch
import random

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any
from scapy.all import *


import titli.fe.corClust as CC
from titli.fe import AfterImage, NetStat

import matplotlib.pyplot as plt

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class BaseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.norm_max = np.ones((input_size,)) * -np.Inf
        self.norm_min = np.ones((input_size,)) * np.Inf
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.sigmoid(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x
    
    def normalise(self, x: np.ndarray, train_mode=False) -> np.ndarray:
        if train_mode:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max -
                                    self.norm_min + 0.0000000000000001)
        return x


class PyTorchKitsune(nn.Module):

    def __init__(self, num_features: int, **kwargs):
        self.state = NetStat()
        self.fe = AfterImage(state=self.state)
        self.hidden_rate = 0.75
        self.FMgrace_rate = 0.2
        self.train_mode = False
        self.num_features = num_features
        super().__init__(**kwargs)
        self.FM = CC.corClust(self.num_features)

    def _define_model(self) -> nn.Module:

        self.tails = nn.ModuleList(
            [BaseAutoencoder(len(c), int(np.ceil(len(c) * self.hidden_rate))) for c in self.clusters]
        )

        for tail in self.tails:
            input_size = len(tail.norm_max)
            a = 1. / input_size

            # Initialize the weights uniformly between -a and a
            nn.init.uniform_(tail.encoder.weight, -a, a)
            
            # Set the decoder weights as the transpose of the encoder weights
            tail.decoder.weight.data = tail.encoder.weight.data.t()

            # Initialize the biases to 0
            nn.init.zeros_(tail.encoder.bias)
            nn.init.zeros_(tail.decoder.bias)
        
        self.head = BaseAutoencoder(
            len(self.clusters), int(np.ceil(len(self.clusters) * self.hidden_rate))
        )
        # Get the input size for the head
        input_size = len(self.head.norm_max)
        a = 1. / input_size

        # Initialize the weights of the head uniformly between -a and a
        nn.init.uniform_(self.head.encoder.weight, -a, a) 
        
        # Set the decoder weights as the transpose of the encoder weights
        self.head.decoder.weight.data = self.head.encoder.weight.data.t()
        nn.init.zeros_(self.head.encoder.bias)
        nn.init.zeros_(self.head.decoder.bias)

    def normalise(self, x: Tensor, train_mode=False) -> Tensor:
        # make a copy of x and convert to numpy
        x_alias = x.clone().detach().cpu().numpy()
        if train_mode:
            self.norm_max[x_alias > self.norm_max] = x_alias[x_alias > self.norm_max]
            self.norm_min[x_alias < self.norm_min] = x_alias[x_alias < self.norm_min]

        # 0-1 normalize
        x = (x - torch.tensor(self.norm_min)) / (torch.tensor(self.norm_max) -
                                    torch.tensor(self.norm_min) + 0.0000000000000001)
        return x
    
    def forward(self, x):
        x = x.reshape(-1, self.num_features)

        x_clusters = []
        for c in self.clusters:
            x_cluster = x[:, c]
            x_clusters.append(x_cluster)

        tail_losses = []
        for tail, c in zip(self.tails, x_clusters):
            c = tail.normalise(c.flatten(), train_mode=self.train_mode)
            # reshape convert to tensor, and pass to device
            c = torch.tensor(c).unsqueeze(0).float().to(self.device) # TODO: This has to be tensor from the start!!!!!!!!!!!!!!!
            output = tail(c)
            if self.train_mode: # TODO: Confirm that this is not Log in training mode!!!!!!!!!!!!
                # loss = torch.log(self.rmse(output, c))
                loss = self.rmse(output, c)
            else:
                loss = self.rmse(output, c)
            if loss.data == 0:
                loss.data = torch.tensor(1e-2)
            tail_losses.append(loss)

        tails = torch.stack(tail_losses)
        tails = self.normalise(tails, train_mode=self.train_mode).float()
        x_hat = self.head(tails)

        return x_hat, tails
    
    def train(self, pcap_path: str) -> None:
        self.train_rmse = []
        self.print_interval = 100
        self.train_mode = True
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.rmse = RMSELoss()
        self.FMgrace = np.floor(self.FMgrace_rate * len(rdpcap(pcap_path)))
        self.dataset = pcap_path.split("/")[-1].split(".")[0]
        
        print("Training Feature Mapping (FM) phase")
        for i, packet in enumerate(PcapReader(pcap_path)):

            # get feature vector
            traffic_vector = self.fe.get_traffic_vector(packet)

            # extract features
            x = self.fe.update(traffic_vector)

            # check for FMgrace period
            if i < self.FMgrace:
                self.FM.update(x)
                continue

            if i == self.FMgrace:
                self.clusters = self.FM.cluster(maxClust=10)
                self._define_model()
                optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
                self.norm_max = np.ones((len(self.clusters),)) * -np.Inf
                self.norm_min = np.ones((len(self.clusters),)) * np.Inf
                print("Training Autoencoder phase")
                continue

            # forward pass
            x_hat, tails = self(x)

            # loss
            loss = self.rmse(x_hat, tails)

            self.train_rmse.append(loss.data)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # # Making the decoder weight the transpose of the encoder weight
            # for tail in self.tails:
            #     tail.decoder.weight.data = tail.encoder.weight.data.t()
            # self.head.decoder.weight.data = self.head.encoder.weight.data.t()

            # Making the encoder weight the transpose of the decoder weight
            for tail in self.tails:
                tail.encoder.weight.data = tail.decoder.weight.data.t()
            self.head.encoder.weight.data = self.head.decoder.weight.data.t()

            if (i + 1) % self.print_interval == 0:
                print(f"Packet: {i} | Loss: {loss.data}")
        
        self.threshold = self.get_threshold(pcap_path)

        return self.threshold
    
    def get_threshold(self, pcap_path: str) -> float:
        print("Calculating threshold")
        self.threshold_rmse = []
        self.print_interval = 10
        self.train_mode = False
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.rmse = RMSELoss()
        
        for _, packet in enumerate(PcapReader(pcap_path)):
            
            # get feature vector
            traffic_vector = self.fe.get_traffic_vector(packet)

            # extract features
            x = self.fe.update(traffic_vector)

            # forward pass
            x_hat, tails = self(x)

            # loss
            loss = self.rmse(x_hat, tails)
            if loss.data == 0:
                loss.data = torch.tensor(1e-2)
            self.threshold_rmse.append(loss.data)

        log_re = np.log(self.threshold_rmse)
        mean = np.mean(log_re)
        std = np.std(log_re)
        threshold_std = np.exp(mean + 3 * std)
        threshold_max = max(self.threshold_rmse)
        threshold = min(threshold_max, threshold_std)

        return threshold

    def infer(self, pcap_path: str) -> None:
        print("Inferencing")
        self.rmse_array = []
        self.print_interval = 10
        self.train_mode = False
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.rmse = RMSELoss()
        
        for i, packet in enumerate(PcapReader(pcap_path)):
            
            # get feature vector
            traffic_vector = self.fe.get_traffic_vector(packet)

            # extract features
            x = self.fe.update(traffic_vector)

            # forward pass
            x_hat, tails = self(x)

            # loss
            loss = self.rmse(x_hat, tails)
            if loss.data == 0:
                loss.data = torch.tensor(1e-2)
            self.rmse_array.append(loss.data)

            if (i + 1) % self.print_interval == 0:
                print(f"Inferencing: {i} | Loss: {loss.data}")

    def plot_kitsune(self, threshold, out_image = "kitsune_plot.png"):
        _ = plt.get_cmap('Set3')

        f, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=600)
        x_val = np.arange(len(self.rmse_array))

        try:
            ax1.scatter(x_val, self.rmse_array, s=1, c='#00008B')
        except:
            ax1.scatter(x_val, self.rmse_array, s=1, alpha=1.0, c='#FF8C00')

        ax1.axhline(y=threshold, color='b', linestyle='-')
        ax1.set_yscale("log")
        ax1.set_title(f"Anomaly Scores of Kitsune Execution Phase: {self.dataset}")
        ax1.set_ylabel("RMSE")
        ax1.set_xlabel("Packet Index")

        f.savefig(out_image)
        print("plot path:", out_image)
        plt.close()
