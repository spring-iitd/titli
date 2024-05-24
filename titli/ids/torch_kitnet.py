import torch
import pickle
import numpy as np
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class Autoencoder(nn.Module):
    def __init__(self, W, hbias, vbias):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(W.shape[0], W.shape[1], bias=True)
        self.encoder.weight.data = torch.from_numpy(W.T)
        self.encoder.bias.data = torch.from_numpy(hbias)
        self.decoder = nn.Linear(W.shape[1], W.shape[0], bias=True)
        self.decoder.weight.data = torch.from_numpy(W)
        self.decoder.bias.data = torch.from_numpy(vbias)
        self.rmse = RMSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class _TorchKitNET(nn.Module):
    def __init__(self, tail_weights, head_weight, clusters, num_features):
        super(_TorchKitNET, self).__init__()

        self.clusters = clusters
        # add the tails of the autoencoders
        self.tails = nn.ModuleList([Autoencoder(weight['W'], weight['hbias'], weight['vbias']) for weight in tail_weights])

        # add the head of the autoencoder
        self.head = Autoencoder(head_weight['W'], head_weight['hbias'], head_weight['vbias'])

        self.num_features = num_features

    def forward(self, x):
        x = x.view(-1, self.num_features)

        x_clusters = [
            torch.index_select(x, 1, torch.tensor(c)) for c in self.clusters
        ]

        tail_losses = []
        for tail, c in zip(self.tails, x_clusters):
            output = tail(c)
            print("I'm never getting executed, if you see me, log shouldn't be here!")
            loss = torch.log(self.rmse(output, c))
            tail_losses.append(loss)
        
        tails = torch.stack(tail_losses)
        x = self.head(tails)

        return x, tails


class BaseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.sigmoid(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x


class TorchKitNET(nn.Module):
    def __init__(self, clusters: list, norms_path: str):
        super(TorchKitNET, self).__init__()
        self.dataset = "PcapDatasetRaw"
        self.input_dim = sum([len(c) for c in clusters])
        self.hr = 0.75
        self.clusters = clusters
        self.rmse = RMSELoss()
        self.tails = nn.ModuleList([BaseAutoencoder(len(c), int(np.ceil(len(c) * self.hr))) for c in clusters])
        self.head = BaseAutoencoder(len(clusters), int(np.ceil(len(clusters) * self.hr)))
        with open(norms_path, "rb") as f:
            self.norm_params = pickle.load(f)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x_clusters = []
        for c in self.clusters:
            norm_max = torch.tensor(self.norm_params[f"norm_max_{c[0]}"]).to(x.device)
            norm_min = torch.tensor(self.norm_params[f"norm_min_{c[0]}"]).to(x.device)

            x_cluster = torch.index_select(x, 1, torch.tensor(c).to(x.device))
            x_cluster = (x_cluster - norm_min) / (norm_max - norm_min + 0.0000000000000001)
            x_cluster = x_cluster.float()

            x_clusters.append(x_cluster)

        tail_losses = []
        for tail, c in zip(self.tails, x_clusters):

            output = tail(c)
            loss = self.rmse(output, c)
            if loss.data == 0:
                loss.data = torch.tensor(1e-2)

            tail_losses.append(loss)

        tails = torch.stack(tail_losses)

        # nomalize the tails
        norm_max = torch.tensor(self.norm_params["norm_max_output"]).to(x.device)
        norm_min = torch.tensor(self.norm_params["norm_min_output"]).to(x.device)
        tails = (tails - norm_min) / (norm_max - norm_min + 0.0000000000000001)
        tails = tails.float()
        x = self.head(tails)

        return x, tails

