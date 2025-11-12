from .base_ids import BaseSKLearnModel
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, confusion_matrix, 
                             accuracy_score, roc_curve, auc)

from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class LOF(BaseSKLearnModel):
    """
    Local Outlier Factor (LOF) model for anomaly detection in NIDS. Adapted from the 
    paper: http://dx.doi.org/10.1137/1.9781611972733.3
    """
    def __init__(self,dataset_name=None, input_size=None, device=None):
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=20, contamination="auto", novelty=True)
        self.model_name = self.__class__.__name__
        super().__init__(dataset_name,input_size,device)

    def __name__(self):
        return "LOF"


if __name__ == "__main__":
    batch_size = 32
    model = LOF()

    data = pd.read_csv("../../utils/weekday_20k.csv")
    model.dataLoader(data)
    model.train_model()
    model.save_model(f"{model.__class__.__name__}_model.pkl")
    model.load_model(f"{model.__class__.__name__}_model.pkl")
    results = model.infer()

    results = model.evaluate(results["y_test"], results["y_pred"])
    model.plot(results)

    from pprint import pprint
