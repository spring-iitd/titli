from .base_ids import BaseSKLearnModel

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class OCSVM(BaseSKLearnModel):
    def __init__(self, dataset_name=None, input_size=None, device=None, titles=None):
        self.title = titles
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=0.01)
        self.model_name = self.__class__.__name__
        super().__init__(dataset_name, input_size, device)

    def __name__(self):
        return "OCSVM"


if __name__ == "__main__":
    batch_size = 32
    model = OCSVM()

    # Load your data
    data = pd.read_csv("/home/subrat/Projects/titli/utils/weekday_20k.csv")
    
    # data = pd.read_csv("/home/kundan/byte-me/data/cic_csv/cic-2023_chopped/Benign_Final/BenignTraffic.csv")
    
    # feature, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
    # dataset = TensorDataset(torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
    
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model.dataLoader(data)
    # Train the model (which will fit the scaler)
    model.train_model()
    model.save_model(f"{model.__class__.__name__}_model.pkl")
    model.load_model(f"{model.__class__.__name__}_model.pkl")

    # Assuming you have an infer method for testing (you can implement it as per your need)
    results = model.infer()

    results = model.evaluate(results["y_test"], results["y_pred"])
    model.plot(results)

    # Compute ROC and save the plot
    model.compute_roc()

    from pprint import pprint
    pprint(results)
