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
    def __init__(self,dataset_name=None, input_size=None, device=None,titles = None):
        self.title = titles
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=20, contamination="auto", novelty=True)
        self.model_name = self.__class__.__name__
        super().__init__(dataset_name,input_size,device)

    def __name__(self):
        return "LOF"
    
    # def evaluate(self, y_test, y_pred, reconstruction_errors=None):
    #     # Compute confusion matrix
    #     cm = confusion_matrix(y_test, y_pred)

    #     # Compute evaluation metrics
    #     f1 = round(f1_score(y_test, y_pred, zero_division=1), 3)
    #     precision = round(precision_score(y_test, y_pred, zero_division=1), 3)
    #     recall = round(recall_score(y_test, y_pred, zero_division=1), 3)
    #     accuracy = round(accuracy_score(y_test, y_pred), 3)

    #     results = {
    #         "f1": f1,
    #         "precision": precision,
    #         "recall": recall,
    #         "accuracy": accuracy,
    #         "confusion_matrix": cm
    #     }
    #      # Print the evaluation metrics
    #     print(f"F1 Score: {f1}")
    #     print(f"Precision: {precision}")
    #     print(f"Recall: {recall}")
    #     print(f"Accuracy: {accuracy}")
    #     metrics =f"./artifacts/{self.dataset_name}/objects/metrics/{self.model_name.lower()}"+"_"+str(self.title)+".txt"

    #     with open(metrics, "w") as file:
    #         file.write(f"F1 Score: {f1}\n")
    #         file.write(f"Precision: {precision}\n")
    #         file.write(f"Recall: {recall}\n")
    #         file.write(f"Accuracy: {accuracy}\n")
         
    #     self.plot(results)
    
    
    def evaluate(self, y_test, y_pred, reconstruction_errors=None):
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Compute evaluation metrics
        f1 = round(f1_score(y_test, y_pred, zero_division=1), 3)
        precision = round(precision_score(y_test, y_pred, zero_division=1), 3)
        recall = round(recall_score(y_test, y_pred, zero_division=1), 3)  # TPR
        accuracy = round(accuracy_score(y_test, y_pred), 3)

        # Derived metrics
        tpr = recall
        fnr = round(fn / (fn + tp), 3) if (fn + tp) else 0.0
        fpr_val = round(fp / (fp + tn), 3) if (fp + tn) else 0.0
        tnr = round(tn / (tn + fp), 3) if (tn + fp) else 0.0

        results = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "confusion_matrix": cm
        }

        # Print metrics
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall (TPR): {recall}")
        print(f"Accuracy: {accuracy}")

        # Save metrics to file
        metrics_path = f"./artifacts/{self.dataset_name}/objects/metrics/{self.model_name.lower()}_{self.title}.txt"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as file:
            file.write(f"Accuracy:       {accuracy:.3f}\n")
            file.write(f"Precision:      {precision:.3f}\n")
            file.write(f"Recall (TPR):   {tpr:.3f}\n")
            file.write(f"F1 Score:       {f1:.3f}\n")
            file.write("\nConfusion Matrix:\n")
            file.write(f"TP: {tp}\n")
            file.write(f"TN: {tn}\n")
            file.write(f"FP: {fp}\n")
            file.write(f"FN: {fn}\n")
            file.write(f"TPR (Recall):   {tpr:.3f}\n")
            file.write(f"FNR:            {fnr:.3f}\n")
            file.write(f"FPR:            {fpr_val:.3f}\n")
            file.write(f"TNR:            {tnr:.3f}\n")

        # Call plotting function
        self.plot(results)



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
