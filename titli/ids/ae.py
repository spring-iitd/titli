# from .base_ids import PyTorchModel
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset

# import argparse
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler

# class Autoencoder(PyTorchModel):

#     def __init__(self, dataset_name, input_size, device,titles):
#         self.title = titles
#         super().__init__(dataset_name, input_size, device)
#         self.criterion = nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

#     def get_model(self):
#         return nn.Sequential(
#             nn.Linear(self.input_size, 8),
#             nn.ReLU(),
#             nn.BatchNorm1d(8),
#             nn.Linear(8, 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(2),
#             nn.Linear(2, 8),
#             nn.ReLU(),
#             nn.Linear(8, self.input_size),
#             # nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)


from .base_ids import PyTorchModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class Autoencoder(PyTorchModel):
    def __init__(self, dataset_name, input_size, device, titles):
        self.title = titles
        super().__init__(dataset_name, input_size, device)

        # Now we use encoder and decoder instead of self.model directly
        self.encoder = self.get_encoder().to(self.device)
        self.decoder = self.get_decoder().to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def get_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 8),
            nn.ReLU(),
            # nn.BatchNorm1d(8),
            nn.Linear(8, 2),
            # nn.ReLU(),
            # nn.BatchNorm1d(2),
        )

    def get_decoder(self):
        return nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, self.input_size),
            # nn.Sigmoid()  # Optional for 0-1 data
        )

    def get_model(self):
        # This satisfies the base class requirement
        return nn.Sequential(
            self.get_encoder(),
            self.get_decoder()
        )

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train and evaluate Autoencoder model")
    # # parser.add_argument("--data-path", type=str, default="/home/kundan/byte-me/data/cic_csv/final_output.csv", help="Path to the dataset")

    # parser.add_argument("--data-path", type=str, default="/home/kundan/byte-me/data/cic_csv/cic-2023_chopped/Benign_Final/BenignTraffic.csv", help="Path to the dataset")
    # parser.add_argument("--model-path", type=str, default="autoencoder.pth", help="Path to save the trained model")
    # parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    # parser.add_argument("--device", type=str, default="cpu", help="Device to use for training and evaluation")
    # args = parser.parse_args()
    
    # model_path = args.model_path
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # model = Autoencoder(input_size=100, device=device).to(device)

    # data = pd.read_csv(args.data_path)
    # feature, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
    
    # scaler = StandardScaler()
    # feature = scaler.fit_transform(feature)
    # dataset = TensorDataset(torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # model.train_model(dataloader)
    # model.save(model_path)
    # model.load(Autoencoder, model_path)
    # y_test,y_pred = model.infer(dataloader)
    # model.evaluate(y_test,y_pred)

    # Dataset
    batch_size = 32
    data = pd.read_csv("../../utils/weekday_20k.csv")

    feature, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
    dataset = TensorDataset(torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Dataset
    ids = Autoencoder(input_size=100, device="cpu")
    # ids.train_model(dataloader)
    # ids.save("autoencoder")
