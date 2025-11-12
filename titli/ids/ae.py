from .base_ids import PyTorchModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class Autoencoder(PyTorchModel):
    def __init__(self, dataset_name, input_size, device):
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
    
    def evaluate(self, test_loader):
        """Comprehensive evaluation: compute metrics, generate plots, and save results.
        
        This method performs complete model evaluation including:
        - Computing all metrics (F1, Precision, Recall, Accuracy, AUC)
        - Generating confusion matrix and ROC curve plots
        - Saving metrics to file
        
        For just getting predictions without metrics, use infer() instead.
        
        Args:
            test_loader (DataLoader): DataLoader containing test data
            
        Returns:
            None: Metrics and plots are saved to disk
        """
        print("Running Autoencoder evaluation...")
        
        # Use infer to get predictions
        y_test, y_pred, reconstruction_errors = self.infer(test_loader)
        
        # Convert to numpy arrays if they aren't already
        y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
        y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        
        # Ensure arrays are 1D
        if y_test.ndim > 1:
            y_test = y_test.ravel()
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        
        print(f"Evaluated {len(y_test)} samples")
        print(f"Threshold: {self.threshold:.6f}")
        
        self.plot_anomaly(reconstruction_errors)
        # Call parent evaluate method (computes metrics and generates plots)
        super().evaluate(y_test, y_pred, reconstruction_errors)


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

    # Initialize and train model
    dataset_name = "test-autoencoder"
    model = Autoencoder(dataset_name=dataset_name, input_size=100, device="cpu")
    
    # Option 1: Train and evaluate
    model.train_model(dataloader)
    model.save()
    
    # Option 2: Complete evaluation (recommended for most users)
    print("\n=== Complete Evaluation ===")
    model.evaluate(dataloader)
    
    # Option 3: Just get predictions (for custom workflows)
    print("\n=== Custom Workflow ===")
    y_true, y_pred, errors = model.infer(dataloader)
    model.plot_anomaly(errors)  # Custom anomaly score plot
    print(f"Custom analysis: Anomaly rate = {y_pred.sum() / len(y_pred):.2%}")

