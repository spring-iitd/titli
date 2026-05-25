import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ._compat import get_pytorch_model_base

PyTorchModel = get_pytorch_model_base()


class Autoencoder(PyTorchModel):
    """
    Drop-in replacement for titli.ids.Autoencoder with a deeper architecture
    and per-sample threshold calibration.

    Architecture: input -> 64 -> 32 -> 16 (latent) -> 32 -> 64 -> input
    """

    def __init__(self, dataset_name, input_size, device,
                 latent_dim=16, hidden_1=64, hidden_2=32,
                 lr=1e-3, weight_decay=1e-5, dropout=0.0,
                 threshold_percentile=95.0):
        self.latent_dim = latent_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.dropout = dropout
        self.threshold_percentile = threshold_percentile

        super().__init__(dataset_name, input_size, device)

        self.encoder = self.get_encoder().to(self.device)
        self.decoder = self.get_decoder().to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def get_encoder(self):
        layers = [
            nn.Linear(self.input_size, self.hidden_1),
            nn.BatchNorm1d(self.hidden_1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers += [
            nn.Linear(self.hidden_1, self.hidden_2),
            nn.BatchNorm1d(self.hidden_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_2, self.latent_dim),
        ]
        return nn.Sequential(*layers)

    def get_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_2),
            nn.BatchNorm1d(self.hidden_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_2, self.hidden_1),
            nn.BatchNorm1d(self.hidden_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_1, self.input_size),
        )

    def get_model(self):
        return nn.Sequential(self.get_encoder(), self.get_decoder())

    def forward(self, x):
        x = x.to(self.device)
        return self.decoder(self.encoder(x))

    def train_model(self, train_loader, epochs=None):
        if epochs is None:
            epochs = self.epochs

        all_train_data = [inputs.numpy() for inputs, _ in train_loader]
        all_train_data = np.concatenate(all_train_data, axis=0)
        self.scaler.fit(all_train_data)

        for epoch in range(epochs):
            self.train()
            running_loss, batch_count = 0.0, 0
            for inputs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)

                self.optimizer.zero_grad()
                outputs = self(inputs_scaled)
                loss = self.criterion(outputs, inputs_scaled)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / batch_count if batch_count else float("inf")
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
            self.scheduler.step(avg_loss)

        self.calculate_threshold(train_loader)

    def calculate_threshold(self, train_loader):
        """Per-sample threshold at the configured percentile of benign reconstruction errors."""
        print("Calculating per-sample threshold on training data...")
        self.eval()
        per_sample_errors = []
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Calculating threshold"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                outputs = self(inputs_scaled)
                errors = (outputs - inputs_scaled).pow(2).mean(dim=1).cpu().numpy()
                per_sample_errors.extend(errors)

        per_sample_errors = np.array(per_sample_errors)
        self.threshold = float(np.percentile(per_sample_errors, self.threshold_percentile))
        qs = np.percentile(per_sample_errors, [50, 75, 90, 95, 99, 99.5, 99.9])
        print(
            f"Threshold (p{self.threshold_percentile}): {self.threshold:.6f}\n"
            f"  mean: {per_sample_errors.mean():.6f} | "
            f"std: {per_sample_errors.std():.6f} | "
            f"max: {per_sample_errors.max():.6f}\n"
            f"  p50={qs[0]:.4f} p75={qs[1]:.4f} p90={qs[2]:.4f} "
            f"p95={qs[3]:.4f} p99={qs[4]:.4f} p99.5={qs[5]:.4f} p99.9={qs[6]:.4f}"
        )

    def evaluate(self, test_loader):
        print("Running Autoencoder evaluation...")
        y_test, y_pred, reconstruction_errors = self.infer(test_loader)

        y_test = np.array(y_test).ravel()
        y_pred = np.array(y_pred).ravel()

        print(f"Evaluated {len(y_test)} samples")
        print(f"Threshold: {self.threshold:.6f}")

        self.plot_anomaly(reconstruction_errors)
        super().evaluate(y_test, y_pred, reconstruction_errors)
