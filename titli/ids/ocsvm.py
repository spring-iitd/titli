from base_ids import BaseSKLearnModel

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
    def __init__(self):
        scaler = StandardScaler()
        model = OneClassSVM(nu=0.01)
        super().__init__(scaler,model)

    def __name__(self):
        return "OCSVM"
    
    def compute_roc(self, test_loader):
        """
        Compute ROC Curve and AUC using the fitted model and scaler.
        """
        X_test, y_test = [], []
        device = "cpu"
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            X_test.append(inputs.cpu().numpy())
            y_test.append(labels.cpu().numpy())

        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        # Transform the test data using the fitted scaler
        # X_test_scaled = self.scaler.transform(X_test)

        # Get decision scores
        scores = self.model.decision_function(X_test)

        # Binary prediction
        y_pred = np.where(scores >= 0, 0, 1)  # 0=Normal, 1=Anomaly

        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")

        # EER Calculation
        fnr = 1 - tpr
        eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer_threshold = thresholds[eer_threshold_index]
        eer = fpr[eer_threshold_index]
        print(f"EER: {eer:.4f}")

        # Save ROC Plot with EER annotation
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.scatter(fpr[eer_threshold_index], tpr[eer_threshold_index], color='red', label=f'EER = {eer:.2f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.title("ROC Curve")
        plt.grid(True)
        roc_filename = f"roc_curve.png"  # You can modify this with actual names if needed
        plt.savefig(roc_filename)
        plt.close()
        print(f"ROC Curve saved to {roc_filename}")
    

if __name__ == "__main__":
    batch_size = 32
    model = OCSVM()

    # Load your data
    data = pd.read_csv("/home/kundan/byte-me/data/cic_csv/final_output.csv")

    # data = pd.read_csv("/home/kundan/byte-me/data/cic_csv/cic-2023_chopped/Benign_Final/BenignTraffic.csv")
    
    feature, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
    dataset = TensorDataset(torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model (which will fit the scaler)
    # model.train(dataloader)
    # model.save_model(f"{model.__class__.__name__}_model.pkl")
    model.load_model(f"{model.__class__.__name__}_model.pkl")

    # Assuming you have an infer method for testing (you can implement it as per your need)
    y_test, y_pred = model.infer(dataloader)

    # Evaluate the model
    results = model.evaluate(y_test, y_pred)
    model.plot(results)

    # Compute ROC and save the plot
    model.compute_roc(dataloader)

    from pprint import pprint
    pprint(results)
