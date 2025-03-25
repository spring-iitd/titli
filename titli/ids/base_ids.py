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
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter


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
            
   

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump((self.scaler, self.model), f)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.scaler, self.model = pickle.load(f)



class PyTorchModel(nn.Module):
    def __init__(self, model_name, input_size, device):
        super(PyTorchModel, self).__init__()
        self.model_name = model_name
        self.device = device
        self.input_size = input_size
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.model = self.get_model()
        self.threshold = 0.0

    def get_model(self):
        """
        Abstract method to be overridden by specific model classes.
        """
        raise NotImplementedError("Must be implemented by subclass")
    
    def train_model(self, train_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(1):  # You can adjust the number of epochs
            running_loss = 0.0
            for inputs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        self.calculate_threshold(train_loader)
        

    def save(self, model_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler
        }, model_path)
        print(f"Model saved to {model_path}")


    def load(self,model_class, model_path):
        # Create model instance with passed input_size and device
        model = model_class(input_size=self.input_size, device = self.device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint


    def calculate_threshold(self,train_loader):
        self.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Calculating threshold"):
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, inputs)
                reconstruction_errors.append(loss.item())

        self.threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        print(f"Threshold: {self.threshold}")
        threshold_file = "threshold.pkl"
        pickle.dump(self.threshold, open("threshold.pkl", 'wb')); print(f"Threshold saved to {threshold_file}")
    

    
    def infer(self, test_loader):
        """
        Infers on the test set and returns the true labels and predicted labels.
        """
        with open("threshold.pkl", 'rb') as f:
            threshold = pickle.load(f)
        print("Using the threshold of {:.2f}".format(threshold))
        self.eval()
        reconstruction_errors = []
        y_test = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Inferencing"):
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, inputs)

                sample_reconstruction_errors = (outputs - inputs).pow(2).mean(dim=1).cpu().numpy()  # per-sample error
                reconstruction_errors.extend(sample_reconstruction_errors)
                y_test.extend(labels.cpu().numpy())

                # Apply threshold to each sample's reconstruction error and create binary prediction
                y_pred.extend((sample_reconstruction_errors > self.threshold).astype(int))

        return y_test , y_pred , reconstruction_errors
    def evaluate(self,y_test, y_pred, reconstruction_errors,test_loader, device="cpu", cm_save_path="confusion_matrix.png", roc_save_path="roc_curve.png"):
        """
        Evaluates the model on the test set, calculates evaluation metrics, and plots confusion matrix and ROC curve.
        """
        with open("threshold.pkl", 'rb') as f:
            threshold = pickle.load(f)
        print("Using the threshold of {:.2f}".format(threshold))
        model = self.model
        model.eval()
    

        # with torch.no_grad():
        #     for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        #         inputs = inputs.to(device)
        #         outputs = model(inputs)
        #         loss = self.criterion(outputs, inputs)

        #         # Compute the per-sample reconstruction error (assuming MSELoss)
        #         sample_reconstruction_errors = (outputs - inputs).pow(2).mean(dim=1).cpu().numpy()  # per-sample error
        #         reconstruction_errors.extend(sample_reconstruction_errors)
        #         y_test.extend(labels.cpu().numpy())

        #         # Apply threshold to each sample's reconstruction error and create binary prediction
        #         y_pred.extend((sample_reconstruction_errors > self.threshold).astype(int))

        # Ensure lengths match

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Compute evaluation metrics
        f1 = round(f1_score(y_test, y_pred, zero_division=1), 3)
        precision = round(precision_score(y_test, y_pred, zero_division=1), 3)
        recall = round(recall_score(y_test, y_pred, zero_division=1), 3)
        accuracy = round(accuracy_score(y_test, y_pred), 3)

        # Print the evaluation metrics
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")

        # --- Confusion Matrix Plot ---
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt=",.0f", cmap="Blues", 
                        xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"], 
                        cbar=True)  # Add color bar explicitly

        # Format the annotations to use scientific notation
        for text in ax.texts:
            # Remove commas and convert to float, then format as scientific notation
            text_value = text.get_text().replace(',', '')  # Remove commas
            try:
                text.set_text(f'{float(text_value):.2e}')  # Format as scientific notation (e.g., 2.55E+05)
            except ValueError:
                continue  # Skip any annotation that can't be converted to float (e.g., if it's empty or NaN)

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(cm_save_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_save_path}")

        # --- ROC Curve and EER Calculation ---
        if np.sum(y_test) == 0 or np.sum(y_test) == len(y_test):
            print("Warning: ROC curve cannot be computed because y_test contains only one class.")
        else:
            fpr, tpr, thresholds = roc_curve(y_test, reconstruction_errors)
            roc_auc = auc(fpr, tpr)

            eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
            eer_threshold = thresholds[eer_index]
            eer = fpr[eer_index]

            # --- ROC Curve Plot ---
            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="blue")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.scatter(fpr[eer_index], tpr[eer_index], color='red', label=f"EER = {eer:.3f} at Threshold = {eer_threshold:.3f}")
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("ROC Curve with EER")
            plt.legend()
            plt.grid()
            plt.savefig(roc_save_path)
            plt.close()
            print(f"ROC curve saved to {roc_save_path}")

            # Display AUC and EER values in decimal format
            print(f"AUC: {roc_auc:.3f}, EER: {eer:.3f} at threshold {eer_threshold:.3f}")
