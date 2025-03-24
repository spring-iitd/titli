from .base_ids import BaseSKLearnModel

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class LOF(BaseSKLearnModel):

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)

    def __name__(self):
        return "LOF"

if __name__ == "__main__":
    batch_size = 32
    model = LOF()

    data = pd.read_csv("../../utils/weekday_20k.csv")
    
    feature, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
    dataset = TensorDataset(torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train(dataloader)
    model.save_model(f"{model.__class__.__name__}_model.pkl")
    model.load_model(f"{model.__class__.__name__}_model.pkl")
    y_test, y_pred = model.infer(dataloader)

    results = model.evaluate(y_test, y_pred)
    model.plot(results)

    from pprint import pprint
    pprint(results)
