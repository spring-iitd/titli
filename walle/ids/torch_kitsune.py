import time
from matplotlib import ticker
import torch
import random

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any
from scapy.all import *


import walle.fe.corClust as CC
from walle.fe.after_image import AfterImage, NetStat
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

        # print(self.norm_max.shape, x.shape)
        # import pdb; pdb.set_trace()
        if train_mode:
            # if self.norm_max.dim() != x.dim():
            #     self.norm_max = self.norm_max.view(x.shape)
            # if self.norm_min.dim() != x.dim():
            #     self.norm_min = self.norm_min.view(x.shape)
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max -
                                    self.norm_min + 0.0000000000000001)
        return x


class Kitsune(nn.Module):

    def __init__(self, **kwargs):
        self._feature_mapper()
        self.state = NetStat()
        self.fe = AfterImage(state=self.state)
        self.hidden_rate = 0.75
        self.FMgrace_rate = 0.2
        self.train_mode = False
        super().__init__(**kwargs)
        self.FM = CC.corClust(100)

    def _feature_mapper(self, x: torch.Tensor=None) -> None:
        # self.clusters = [
        #     [21, 28],
        #     [35],
        #     [64, 63, 60, 57, 51, 54, 61, 58, 52, 55],
        #     [71, 78, 85, 92, 99, 84, 91, 98, 70, 77],
        #     [42, 49, 48, 41, 34, 20, 27],
        #     [11, 38, 14, 45],
        #     [67, 74, 81, 88, 95],
        #     [8, 31, 2, 17, 5, 24],
        #     [39, 46, 82, 89, 96, 68, 75, 32, 18, 25],
        #     [12, 43],
        #     [9, 36],
        #     [62, 93],
        #     [59, 86],
        #     [6, 29],
        #     [56, 79, 50, 65, 53, 72, 0, 15, 3, 22],
        #     [66, 73, 80, 87, 94],
        #     [10, 37, 13, 44, 7, 30, 1, 16, 4, 23],
        #     [90, 97, 83, 69, 76, 40, 47, 33, 19, 26],
        # ]
        self.clusters = []

    # def _define_model(self) -> nn.Module:

    #     self.tails = nn.ModuleList(
    #         [BaseAutoencoder(len(c), int(np.ceil(len(c) * self.hidden_rate))) for c in self.clusters]
    #     )

    #     for tail in self.tails:
    #         # get the input size of the tail
    #         input_size = len(tail.norm_max)

    #     self.head = BaseAutoencoder(
    #         len(self.clusters), int(np.ceil(len(self.clusters) * self.hidden_rate))
    #     )
    

    '''Here we have initialized the weights uniformly between -a and a, where a = 1/input_size. and
    we have initialized the biases to 0. We have also set the decoder weights as the transpose of the encoder weights.
    '''
    def _define_model(self) -> nn.Module:

        self.tails = nn.ModuleList(
            [BaseAutoencoder(len(c), int(np.ceil(len(c) * self.hidden_rate))) for c in self.clusters]
        )

        for tail in self.tails:
            input_size = len(tail.norm_max)  # Get the input size
            # print(f"Input size : {input_size}")
            a = 1. / input_size  # Calculate the value of a
            nn.init.uniform_(tail.encoder.weight, -a, a)  # Initialize the weights uniformly between -a and a
            #print the encoder weight
            # print(f"Encoder weight : {tail.encoder.weight}")
            # print(f"Encoder weight dimension : {tail.encoder.weight.shape}")
            tail.decoder.weight.data = tail.encoder.weight.data.t()  # Set the decoder weights as the transpose of the encoder weights
            nn.init.zeros_(tail.encoder.bias)  # Initialize the biases to 0
            nn.init.zeros_(tail.decoder.bias)  # Initialize the biases to 0
            # print(f"Decoder weight : {tail.decoder.weight}")
            # print(f"Decoder weight dimension : {tail.decoder.weight.shape}")
        
        self.head = BaseAutoencoder(
            len(self.clusters), int(np.ceil(len(self.clusters) * self.hidden_rate))
        )
        input_size = len(self.head.norm_max)  # Get the input size for the head
        a = 1. / input_size  # Calculate the value of a for the head
        nn.init.uniform_(self.head.encoder.weight, -a, a)  # Initialize the weights of the head uniformly between -a and a
        # print(f"Encoder weight : {self.head.encoder.weight}")
        # print(f"Encoder weight dimension : {self.head.encoder.weight.shape}")
        self.head.decoder.weight.data = self.head.encoder.weight.data.t()  # Set the decoder weights as the transpose of the encoder weights
        nn.init.zeros_(self.head.encoder.bias)  # Initialize the biases of the head to 0
        nn.init.zeros_(self.head.decoder.bias)  # Initialize the biases of the head to 0



    def normalise(self, x: Tensor, train_mode=False) -> Tensor:
        
        # make a copy of x and convert to numpy
        x_alias = x.clone().detach().cpu().numpy()
        if train_mode:
            # if self.norm_max.dim() != x.dim():
            #     self.norm_max = self.norm_max.view(x.shape)
            # if self.norm_min.dim() != x.dim():
            #     self.norm_min = self.norm_min.view(x.shape)
            self.norm_max[x_alias > self.norm_max] = x_alias[x_alias > self.norm_max]
            self.norm_min[x_alias < self.norm_min] = x_alias[x_alias < self.norm_min]

        # 0-1 normalize
        x = (x - torch.tensor(self.norm_min)) / (torch.tensor(self.norm_max) -
                                    torch.tensor(self.norm_min) + 0.0000000000000001)
        return x
    
    def forward(self, x):
        x = x.reshape(-1, 100)
        # print(x.shape)

        x_clusters = []
        for c in self.clusters:
            # norm_max = torch.tensor(self.norm_params[f"norm_max_{c[0]}"])
            # norm_min = torch.tensor(self.norm_params[f"norm_min_{c[0]}"])

            # x_cluster = torch.index_select(x, 1, torch.tensor(c))
            # x_cluster = (x_cluster - norm_min) / (
            #     norm_max - norm_min + 0.0000000000000001
            # )
            x_cluster = x[:, c]
            # x_cluster = x_cluster.float()
            # print(type(x_cluster))
            # print(x_cluster)
            x_clusters.append(x_cluster)

        tail_losses = []
        for tail, c in zip(self.tails, x_clusters):
            c = tail.normalise(c.flatten(), train_mode=self.train_mode)
            # reshape convert to tensor, and pass to device
            c = torch.tensor(c).unsqueeze(0).float().to(self.device)
            output = tail(c)
            if self.train_mode:
                # loss = torch.log(self.rmse(output, c))
                loss = self.rmse(output, c)
            else:
                loss = self.rmse(output, c)
            if loss.data == 0:
                loss.data = torch.tensor(1e-2)
            tail_losses.append(loss)

        tails = torch.stack(tail_losses)

        # norm_max = torch.tensor(self.norm_params["norm_max_output"])
        # norm_min = torch.tensor(self.norm_params["norm_min_output"])
        # tails = (tails - norm_min) / (norm_max - norm_min + 0.0000000000000001)
        # tails = tails.float()
        tails = self.normalise(tails, train_mode=self.train_mode).float()
        x_hat = self.head(tails)

        return x_hat, tails
    
    def train(self, pcap_path: str) -> None:
        self.re = []
        self.print_interval = 100
        self.train_mode = True
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.rmse = RMSELoss()
        self.FMgrace = np.floor(self.FMgrace_rate * len(rdpcap(pcap_path)))
        
        for i, packet in enumerate(PcapReader(pcap_path)):
            
            # get feature vector
            traffic_vector = self.fe.get_traffic_vector(packet)

            # extract features
            # x = torch.tensor(self.fe.update(traffic_vector)).to(self.device)
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
                continue
            
            # print(f"Packet: {i} | Cluster: {self.clusters}")
            # forward pass
            x_hat, tails = self(x)

            # loss
            loss = self.rmse(x_hat, tails)

            # if(loss.data > 1):
            #     print(f"Packet over 1 : {i} | Loss: {loss.data}")

            self.re.append(loss.data)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Making the decoder weight the transpose of the encoder weight
            for tail in self.tails:
                tail.decoder.weight.data = tail.encoder.weight.data.t()
            self.head.decoder.weight.data = self.head.encoder.weight.data.t()

            if (i + 1) % self.print_interval == 0:
                print(f"Packet: {i} | Loss: {loss.data}")

        benignSample = np.log(self.re)
        mean = np.mean(benignSample)
        std = np.std(benignSample)
        threshold_std = np.exp(mean + 3 * std)
        threshold_max = max(self.re)
        self.threshold = min(threshold_max, threshold_std)
        print(f"Threshold: {self.threshold}")
    
    def infer(self, pcap_path: str) -> None:
        self.rmse_array = []
        self.print_interval = 10
        self.train_mode = False
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.rmse = RMSELoss()
        # re = []
        
        for i, packet in enumerate(PcapReader(pcap_path)):
            
            # get feature vector
            traffic_vector = self.fe.get_traffic_vector(packet)

            # extract features
            x = self.fe.update(traffic_vector)

            # forward pass
            x_hat, tails = self(x)

            # loss
            loss = self.rmse(x_hat, tails)
            self.rmse_array.append(loss.data)

            if (i + 1) % self.print_interval == 0:
                print(f"Inferencing: {i} | Loss: {loss.data}")

        # return re

    def plot_kitsune(self,threshold,plot_with_time = True, meta_file = False, out_image = "kitsune_plot.png"):
        cmap = plt.get_cmap('Set3')

        f, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=200)

        if plot_with_time:
            x_val = range(len(self.rmse_array))
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.85))
            ax1.tick_params(labelrotation=90)
        else:
            x_val = range(len(self.rmse_array))

        if meta_file:
            ax1.scatter(x_val, self.rmse_array, s=1, c='#00008B')
        else:
            ax1.scatter(x_val, self.rmse_array, s=1, alpha=1.0, c='#FF8C00')

        ax1.axhline(y=threshold, color='r', linestyle='-')
        ax1.set_yscale("log")
        ax1.set_title("Anomaly Scores from Kitsune Execution Phase")
        ax1.set_ylabel("RMSE (log scaled)")
        ax1.set_xlabel("packet index")

        f.savefig(out_image)
        print("plot path:", out_image)
        plt.close()
