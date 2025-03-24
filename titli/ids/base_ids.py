import torch
import pickle
import numpy as np

from tqdm import tqdm
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, confusion_matrix, 
                             accuracy_score, roc_curve, auc)

class BaseSKLearnModel:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def fit(self, X_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train)

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

    def train(self, train_loader):
        X_train = []
        for inputs, _ in tqdm(train_loader, desc="Training"):
            # inputs = inputs.to(device)
            X_train.append(inputs.cpu().numpy())
        X_train = np.vstack(X_train)
        self.model.fit(X_train)

    def infer(self, test_loader):
        X_test, y_test = [], []
        for inputs, labels in tqdm(test_loader, desc="Inferencing"):
            # inputs, labels = inputs.to(device), labels.to(device)
            X_test.append(inputs.cpu().numpy())
            y_test.append(labels.cpu().numpy()) #TODO
            # y_test.append(np.ones(labels.shape[0]))
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)
        y_pred = self.model.predict(X_test)
        y_pred = np.where(y_pred == 1, 0, 1)  # Convert LOF output to binary labels

        return y_test, y_pred
    
    def evaluate(self, y_test, y_pred):

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Compute evaluation metrics
        f1 = round(f1_score(y_test, y_pred, zero_division=1), 3)
        precision = round(precision_score(y_test, y_pred, zero_division=1), 3)
        recall = round(recall_score(y_test, y_pred, zero_division=1), 3)
        accuracy = round(accuracy_score(y_test, y_pred), 3)

        results = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "confusion_matrix": cm
        }

        return results
    
    def plot(self, results):

        cm = results["confusion_matrix"]

        plt.figure(figsize=(6, 5))

        # format numbers in scientific notation or with decimals 
        # (e.g., 1e+05 or 123456.78)
        def fmt(x):
            # If value is less than 10,000 show with decimal precision
            if x < 1e4:
                return f"{x:.2f}"
            else:  # Otherwise show in scientific notation
                return f"{x:.2e}"
    
        # Plot heatmap with custom formatting for annotations
        sns.heatmap(cm, annot=True, fmt="", cmap="Blues",
                    xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"],
                    annot_kws={"size": 12},
                    cbar_kws={"format": plt.FuncFormatter(lambda x, _: fmt(x))})  # Format color bar
    
        # Modify annotations inside boxes to custom formatting
        ax = plt.gca()
        for text in ax.texts:
            text_value = float(text.get_text())
            text.set_text(fmt(text_value))
    
        # Labels and Title
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
    
        # Set tick labels properly (optional if needed for axes)
        ax.set_xticklabels(["Benign", "Malicious"])
        ax.set_yticklabels(["Benign", "Malicious"])
        plt.savefig("confusion_matrix.png")
        plt.show()
        plt.close()
        # plt.savefig(cm_save_path)  # Save figure
        # plt.close()
        # print(f"Confusion matrix saved to {cm_save_path}")
            
    # def compute_roc(self, test_loader):
    #     """
    #     Compute ROC Curve and AUC using the fitted model and scaler.
    #     """
    #     X_test, y_test = [], []
    #     device = "cpu"
    #     for inputs, labels in tqdm(test_loader, desc="Evaluating"):
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         X_test.append(inputs.cpu().numpy())
    #         y_test.append(labels.cpu().numpy())

    #     X_test = np.vstack(X_test)
    #     y_test = np.hstack(y_test)

    #     # Transform the test data using the fitted scaler
    #     # X_test_scaled = self.scaler.transform(X_test)

    #     # Get decision scores
    #     scores = self.model.decision_function(X_test)

    #     # Binary prediction
    #     y_pred = np.where(scores >= 0, 0, 1)  # 0=Normal, 1=Anomaly

    #     # ROC Curve and AUC
    #     fpr, tpr, thresholds = roc_curve(y_test, scores, pos_label=1)
    #     roc_auc = auc(fpr, tpr)
    #     print(f"AUC: {roc_auc:.4f}")

    #     # EER Calculation
    #     fnr = 1 - tpr
    #     eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
    #     eer_threshold = thresholds[eer_threshold_index]
    #     eer = fpr[eer_threshold_index]
    #     print(f"EER: {eer:.4f}")

    #     # Save ROC Plot with EER annotation
    #     plt.figure()
    #     plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    #     plt.plot([0, 1], [0, 1], linestyle='--')
    #     plt.scatter(fpr[eer_threshold_index], tpr[eer_threshold_index], color='red', label=f'EER = {eer:.2f}')
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.legend(loc="lower right")
    #     plt.title("ROC Curve")
    #     plt.grid(True)
    #     roc_filename = f"roc_curve.png"  # You can modify this with actual names if needed
    #     plt.savefig(roc_filename)
    #     plt.close()
    #     print(f"ROC Curve saved to {roc_filename}")

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump((self.scaler, self.model), f)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.scaler, self.model = pickle.load(f)
