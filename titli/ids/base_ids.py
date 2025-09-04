import os
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
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

from titli.utils.data import create_directories

class BaseSKLearnModel:
    def __init__(self, dataset_name, input_size, device):
        # self.scaler = scaler
        # self.model = model
        self.dataset_name = dataset_name

        create_directories(dataset_name)

    def __call__(self, *args, **kwds):
        pass #TODO later

    def train_model(self,train_loader):
        X_train = []
        for inputs, _ in tqdm(train_loader, desc="Training"):
            # inputs = inputs.to(device)
            X_train.append(inputs.cpu().numpy())
        X_train = np.vstack(X_train)
        X_train = self.scaler.fit_transform(X_train) # TODO: Check this later
        self.model.fit(X_train)
        self.calculate_threshold(X_train)

    def calculate_threshold(self, X_train):
        """
        This function calculates the threshold based on the LOF scores from the training data.
        We will set the threshold to the 95th percentile of the LOF scores.
        """
        # Get LOF scores for the training set
        if(self.model_name == "LOF"):
            # scores = self.model.negative_outlier_factor_  # LOF model gives negative outlier factor for training set
            scores = -self.model.score_samples(X_train)
        elif(self.model_name == "OCSVM"):
            scores = self.model.decision_function(X_train) 

        # Calculate the threshold at the 95th percentile of LOF scores
        self.threshold = np.percentile(scores, 95)  # Set threshold at 95th percentile
        print(f"Threshold for anomaly detection set at: {self.threshold}")

    def infer(self, test_loader):
        X_test, y_test = [], []
        for inputs, labels in tqdm(test_loader, desc="Inferencing"):
            # inputs, labels = inputs.to(device), labels.to(device)
            X_test.append(inputs.cpu().numpy())
            y_test.append(labels.cpu().numpy()) #TODO
            # y_test.append(np.ones(labels.shape[0]))
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)
        X_test = self.scaler.transform(X_test) # TODO: Check this later but ignore in pull request
       
        if self.model_name == "LOF":
            reconstruction_errors = -self.model.score_samples(X_test)
            # y_pred = self.model.predict(X_test)
            # y_pred = np.where(y_pred == 1, 0, 1)
        else:
            reconstruction_errors = -self.model.decision_function(X_test)
        y_pred = (reconstruction_errors > self.threshold).astype(int)

        return y_test, y_pred, reconstruction_errors
    
        
    # def evaluate(self, y_test, y_pred, reconstruction_errors):
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
    #     device = "cpu"
        
    #     # y_test = np.hstack(y_test)

    #     # Compute ROC and AUC
    #     count=0
    #     for i in reconstruction_errors:
    #         if(i>-.98):
    #             count+=1
    #     print("Number of anomalies detected: ",count)
    #     fpr, tpr, thresholds = roc_curve(y_test, reconstruction_errors, pos_label=1)
    #     roc_auc = auc(fpr, tpr)
    #     print(f"AUC: {roc_auc:.4f}")

    #     # Plot ROC curve
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    #     plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc='lower right')

    #     # Save the plot
    #     roc_save_path = f"./artifacts/{self.dataset_name}/plots/roc/{self.model_name.lower()}_{self.title}.png"
    #     os.makedirs(os.path.dirname(roc_save_path), exist_ok=True)
    #     plt.savefig(roc_save_path)
    #     plt.close()

    #     print(f"ROC curve saved to {roc_save_path}")
    
    
    def evaluate(self, y_test, y_pred, reconstruction_errors):
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
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")

        # Save metrics
        metrics_path = f"./artifacts/{self.dataset_name}/objects/metrics/{self.model_name.lower()}_{self.title}.txt"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as file:
            file.write(f"Accuracy:    {accuracy:.3f}\n")
            file.write(f"Precision:   {precision:.3f}\n")
            file.write(f"Recall(TPR): {tpr:.3f}\n")
            file.write(f"F1 Score:    {f1:.3f}\n")
            file.write("\nConfusion Matrix:\n")
            file.write(f"TP: {tp}\n")
            file.write(f"TN: {tn}\n")
            file.write(f"FP: {fp}\n")
            file.write(f"FN: {fn}\n")
            file.write(f"TPR (Recall): {tpr:.3f}\n")
            file.write(f"FNR:          {fnr:.3f}\n")
            file.write(f"FPR:          {fpr_val:.3f}\n")
            file.write(f"TNR:          {tnr:.3f}\n")

        # Plot confusion matrix via self.plot method
        self.plot(results)

        # Count anomalies above threshold -0.98
        anomaly_count = np.sum(np.array(reconstruction_errors) > -0.98)
        print("Number of anomalies detected: ", anomaly_count)

        # Compute ROC and AUC
        if np.sum(y_test) == 0 or np.sum(y_test) == len(y_test):
            print("Warning: ROC curve cannot be computed because y_test contains only one class.")
            roc_auc = None
        else:
            fpr, tpr, thresholds = roc_curve(y_test, reconstruction_errors, pos_label=1)
            roc_auc = auc(fpr, tpr)
            print(f"AUC: {roc_auc:.4f}")

            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')

            # Save the plot
            roc_save_path = f"./artifacts/{self.dataset_name}/plots/roc/{self.model_name.lower()}_{self.title}.png"
            os.makedirs(os.path.dirname(roc_save_path), exist_ok=True)
            plt.savefig(roc_save_path)
            plt.close()
            print(f"ROC curve saved to {roc_save_path}")

            # Append AUC to metrics file
            with open(metrics_path, "a") as file:
                file.write(f"\nAUC-ROC:      {roc_auc:.4f}\n")


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
        
        cm_save_path=f"./artifacts/{self.dataset_name}/plots/confusion_matrix/{self.model_name.lower()}"+"_"+str(self.title)+".png"
        print(f"confusion matrix saved to {cm_save_path}")

        plt.savefig(cm_save_path)
        plt.close()

    def save(self, model_path=None):
        if not model_path:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pth"
        with open(model_path, 'wb') as f:
            pickle.dump((self.scaler, self.model,self.threshold), f)
        print(f"Model saved to {model_path}")

    def load(self, model_path=None):
        if not model_path:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pth"
        with open(model_path, 'rb') as f:
            self.scaler, self.model, self.threshold = pickle.load(f)
    
    def plot_anomaly(self, anomaly_score):
        # Extract and format the title
        title = self.title

        # Generate indices for the x-axis
        packet_indices = np.arange(len(anomaly_score))

        # Define color mapping: 0 -> violet, 1 -> red
        # color_mapping = {0: 'black', 1: 'red'}
        # colors = [color_mapping[adv_packet[1]] for adv_packet in adv_packets]
        # Check if the lengths match
        
        print(f"the threshold being use is {self.threshold}")

        # Plot the adversarial malicious data
        colors = 'red'
        plt.scatter(
        packet_indices,
        anomaly_score,
        c=colors,
        label="Malicious",
        alpha=1,
        s=1.5
        )

        # Plot the threshold line
        plt.axhline(y=self.threshold, color="blue", linestyle="--", label="Threshold")

        # Set the title and labels with appropriate font sizes and bold font
        plt.title(title, fontsize=20, fontweight='bold')
        plt.xlabel("Packet Index", fontsize=15, fontweight='bold')
        plt.ylabel("Anomaly Score", fontsize=15, fontweight='bold')

        # Set the y-axis to log scale
        plt.yscale("log")

        # Increase tick size
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend with increased font size and position it outside the plot area
        plt.legend(fontsize=14, loc='upper right')

        # Use tight layout to prevent overlap
        plt.tight_layout()

        # Define the folder path
        # s=str(args.pcap_path).split("data/")[1].split("/")[0]

        plot_path =  f"./artifacts/{self.dataset_name}/plots/anomaly/{self.model_name.lower()}"+"_"+str(self.title)+".png"

        # folder_path = os.path.dirname(plot_path)

        # # Check if the folder exists
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # Show or save the plot
        plt.savefig(plot_path, dpi=300)
        plt.close()


class PyTorchModel(nn.Module):
    def __init__(self, dataset_name, input_size, device):
        super(PyTorchModel, self).__init__()
        self.model_name = self.__class__.__name__
        self.dataset_name = dataset_name
        self.device = device
        self.input_size = input_size
        self.scaler = StandardScaler()
        self.epochs = 5

        self.model = self.get_model()
        create_directories(dataset_name)

        self.threshold = None

    def get_model(self):
        """
        Abstract method to be overridden by specific model classes.
        """
        raise NotImplementedError("Must be implemented by the subclass")
    
    def train_model(self, train_loader):
        all_train_data = []  # Collect all training data in a list
        for inputs, _ in train_loader:
            all_train_data.append(inputs.numpy())  # Convert tensor to numpy
        all_train_data = np.concatenate(all_train_data, axis=0)
        self.scaler.fit(all_train_data)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs)  # Apply the same scaler used during training
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device) 
                self.optimizer.zero_grad()
                outputs = self(inputs_scaled)
                loss = self.criterion(outputs, inputs_scaled)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        self.calculate_threshold(train_loader)

    def save(self, model_path=None):
        if not model_path:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pth"
        torch.save({
            "model_state_dict": self.state_dict(),
            "threshold": self.threshold,
            "scaler": self.scaler,
        }, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path=None):
        if not model_path:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pth"
        checkpoint = torch.load(model_path, weights_only=False)
        # model = self(self.model_name, input_size=self.input_size, device=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.scaler = checkpoint['scaler']

        return checkpoint

    def calculate_threshold(self, train_loader):
        print("Please ensure that you're using a trained model for calculating the threshold.")
        self.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Calculating threshold"):
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs)  # Apply the same scaler used during training
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device)
                outputs = self(inputs_scaled)
                loss = self.criterion(outputs, inputs_scaled)
                reconstruction_errors.append(loss.item())

        self.threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        print(f"Threshold: {self.threshold}")
        

    def infer(self, test_loader):
        """
        Infers on the test set and returns the true labels and predicted labels.
        """
        # threshold_file = "threshold"+str(self.model_name)+".pkl"
        # with open(threshold_file, 'rb') as f:
        #     threshold = pickle.load(f)

        if not self.threshold:
            print("Threshold not set. Please load or train before inferring.")
            return None

        print("Using the threshold of {:.2f}".format(self.threshold))
        self.eval()
        reconstruction_errors = []
        y_test = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Inferencing"):
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs)  # Apply the same scaler used during training
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device) 
                outputs = self(inputs_scaled)
                loss = self.criterion(outputs, inputs_scaled)
                loss= (outputs - inputs_scaled).pow(2).mean(dim=1).cpu().numpy()  # per-sample error
                reconstruction_errors.extend(loss)
                y_test.extend(labels.cpu().numpy())

                # Apply threshold to each sample's reconstruction error and create binary prediction
                y_pred.extend((loss > self.threshold).astype(int))
        return y_test, y_pred , reconstruction_errors

    # def evaluate(self, y_test, y_pred, reconstruction_errors):
    #     """
    #     Evaluates the model on the test set, calculates evaluation metrics, and plots confusion matrix and ROC curve.
    #     """
    #     cm_save_path=f"./artifacts/{self.dataset_name}/plots/confusion_matrix/{self.model_name.lower()}"+"_"+str(self.title)+".png"
    #     roc_save_path=f"./artifacts/{self.dataset_name}/plots/roc/{self.model_name.lower()}"+"_"+str(self.title)+".png"
    #     threshold = self.threshold
    #     print("Using the threshold of {:.2f}".format(threshold))
    
    #     cm = confusion_matrix(y_test, y_pred,labels=[0, 1])

    #     # Compute evaluation metrics
    #     f1 = round(f1_score(y_test, y_pred, zero_division=1), 3)
    #     precision = round(precision_score(y_test, y_pred, zero_division=1), 3)
    #     recall = round(recall_score(y_test, y_pred, zero_division=1), 3)
    #     accuracy = round(accuracy_score(y_test, y_pred), 3)

    #     # Print the evaluation metrics
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

    #     def fmt(x):
    #         # If value is less than 10,000 show with decimal precision
    #         if x < 1e4:
    #             return f"{x:.2f}"
    #         else:  # Otherwise show in scientific notation
    #             return f"{x:.2e}"
    
    #     # Plot heatmap with custom formatting for annotations
    #     sns.heatmap(cm, annot=True, fmt="", cmap="Blues",
    #                 xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"],
    #                 annot_kws={"size": 12},
    #                 cbar_kws={"format": plt.FuncFormatter(lambda x, _: fmt(x))})  # Format color bar
    
    #     # Modify annotations inside boxes to custom formatting
    #     ax = plt.gca()
    #     for text in ax.texts:
    #         text_value = float(text.get_text())
    #         text.set_text(fmt(text_value))

    #     plt.xlabel("Predicted Labels")
    #     plt.ylabel("True Labels")
    #     plt.title("Confusion Matrix")
    #     plt.savefig(cm_save_path)
    #     plt.close()
    #     print(f"Confusion matrix saved to {cm_save_path}")

    #     # --- ROC Curve and EER Calculation ---
    #     if np.sum(y_test) == 0 or np.sum(y_test) == len(y_test):
    #         print("Warning: ROC curve cannot be computed because y_test contains only one class.")
    #     else:
    #         fpr, tpr, thresholds = roc_curve(y_test, reconstruction_errors)
    #         roc_auc = auc(fpr, tpr)

    #         eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
    #         eer_threshold = thresholds[eer_index]
    #         eer = fpr[eer_index]

    #         # --- ROC Curve Plot ---
    #         plt.figure(figsize=(7, 6))
    #         plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="blue")
    #         plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    #         plt.scatter(fpr[eer_index], tpr[eer_index], color='red', label=f"EER = {eer:.3f} at Threshold = {eer_threshold:.3f}")
    #         plt.xlabel("False Positive Rate (FPR)")
    #         plt.ylabel("True Positive Rate (TPR)")
    #         plt.title("ROC Curve with EER")
    #         plt.legend()
    #         plt.grid()
    #         plt.savefig(roc_save_path)
    #         plt.close()
    #         print(f"ROC curve saved to {roc_save_path}")

    #         # Display AUC and EER values in decimal format
    #         print(f"AUC: {roc_auc:.3f}, EER: {eer:.3f} at threshold {eer_threshold:.3f}")
    
    
    def evaluate(self, y_test, y_pred, reconstruction_errors):
        """
        Evaluates the model on the test set, calculates evaluation metrics, and plots confusion matrix and ROC curve.
        """
        cm_save_path = f"./artifacts/{self.dataset_name}/plots/confusion_matrix/{self.model_name.lower()}_{self.title}.png"
        roc_save_path = f"./artifacts/{self.dataset_name}/plots/roc/{self.model_name.lower()}_{self.title}.png"
        metrics_path = f"./artifacts/{self.dataset_name}/objects/metrics/{self.model_name.lower()}_{self.title}.txt"

        threshold = self.threshold
        print(f"Using the threshold of {threshold:.2f}")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Basic metrics
        f1 = round(f1_score(y_test, y_pred, zero_division=1), 3)
        precision = round(precision_score(y_test, y_pred, zero_division=1), 3)
        recall = round(recall_score(y_test, y_pred, zero_division=1), 3)  # TPR
        accuracy = round(accuracy_score(y_test, y_pred), 3)

        # Derived metrics
        tpr = recall
        fnr = round(fn / (fn + tp), 3) if (fn + tp) else 0.0
        fpr = round(fp / (fp + tn), 3) if (fp + tn) else 0.0
        tnr = round(tn / (tn + fp), 3) if (tn + fp) else 0.0

        # Print summary
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")

        # Save metrics to file
        with open(metrics_path, "w") as file:
            file.write(f"Threshold:   {threshold:.3f}\n")
            file.write(f"Accuracy:    {accuracy:.3f}\n")
            file.write(f"Precision:   {precision:.3f}\n")
            file.write(f"Recall(TPR): {tpr:.3f}\n")
            file.write(f"F1 Score:    {f1:.3f}\n")
            file.write("\nConfusion Matrix:\n")
            file.write(f"TP: {tp}\n")
            file.write(f"TN: {tn}\n")
            file.write(f"FP: {fp}\n")
            file.write(f"FN: {fn}\n")
            file.write(f"TPR (Recall): {tpr:.3f}\n")
            file.write(f"FNR:          {fnr:.3f}\n")
            file.write(f"FPR:          {fpr:.3f}\n")
            file.write(f"TNR:          {tnr:.3f}\n")

        # Plot confusion matrix
        def fmt(x):
            return f"{x:.2f}" if x < 1e4 else f"{x:.2e}"

        sns.heatmap(cm, annot=True, fmt="", cmap="Blues",
                    xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"],
                    annot_kws={"size": 12},
                    cbar_kws={"format": plt.FuncFormatter(lambda x, _: fmt(x))})
        
        ax = plt.gca()
        for text in ax.texts:
            text_value = float(text.get_text())
            text.set_text(fmt(text_value))

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(cm_save_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_save_path}")

        # ROC curve and EER
        if np.sum(y_test) == 0 or np.sum(y_test) == len(y_test):
            print("Warning: ROC curve cannot be computed because y_test contains only one class.")
        else:
            fpr_curve, tpr_curve, thresholds = roc_curve(y_test, reconstruction_errors)
            roc_auc = auc(fpr_curve, tpr_curve)

            eer_index = np.nanargmin(np.abs(fpr_curve - (1 - tpr_curve)))
            eer_threshold = thresholds[eer_index]
            eer = fpr_curve[eer_index]

            plt.figure(figsize=(7, 6))
            plt.plot(fpr_curve, tpr_curve, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="blue")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.scatter(fpr_curve[eer_index], tpr_curve[eer_index], color='red',
                        label=f"EER = {eer:.3f} at Threshold = {eer_threshold:.3f}")
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("ROC Curve with EER")
            plt.legend()
            plt.grid()
            plt.savefig(roc_save_path)
            plt.close()
            print(f"ROC curve saved to {roc_save_path}")
            print(f"AUC: {roc_auc:.3f}, EER: {eer:.3f} at threshold {eer_threshold:.3f}")

            # Append AUC and EER to metrics file
            with open(metrics_path, "a") as file:
                file.write(f"\nAUC-ROC:      {roc_auc:.3f}\n")
                file.write(f"EER:          {eer:.3f}\n")
                file.write(f"EER Threshold:{eer_threshold:.3f}\n")

    
    def plot_anomaly(self, anomaly_score):

      
        plt.figure(figsize=(12, 8))

        # Extract and format the title
        title = self.title

        # Generate indices for the x-axis
        packet_indices = np.arange(len(anomaly_score))
        

        # Plot the adversarial malicious data
        colors = 'red'
        plt.scatter(
        packet_indices,
        anomaly_score,
        c=colors,
        label="Malicious",
        alpha=1,
        s=1.5
    )

        # Plot the threshold line
        plt.axhline(y=self.threshold, color="blue", linestyle="--", label="Threshold")

        # Set the title and labels with appropriate font sizes and bold font
        plt.title(title, fontsize=20, fontweight='bold')
        plt.xlabel("Packet Index", fontsize=15, fontweight='bold')
        plt.ylabel("Anomaly Score", fontsize=15, fontweight='bold')

        # Set the y-axis to log scale
        plt.yscale("log")

        # Increase tick size
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend with increased font size and position it outside the plot area
        plt.legend(fontsize=14, loc='upper right')

        # Use tight layout to prevent overlap
        plt.tight_layout()

        # Define the folder path
        # s=str(args.pcap_path).split("data/")[1].split("/")[0]

        plot_path =  f"./artifacts/{self.dataset_name}/plots/anomaly/{self.model_name.lower()}"+"_"+str(self.title)+".png"

        # folder_path = os.path.dirname(plot_path)

        # # Check if the folder exists
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # Show or save the plot
        plt.savefig(plot_path, dpi=300)
        plt.close()

