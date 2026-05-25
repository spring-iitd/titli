import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ._compat import get_pytorch_model_base

PyTorchModel = get_pytorch_model_base()


class FeatureTokenizer(nn.Module):
    """
    TabTransformer-style: each scalar feature gets its own learned (weight, bias)
    that lifts it to a d_model-dim token.
    """

    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):
        # x: [B, n_features] -> tokens: [B, n_features, d_model]
        return x.unsqueeze(-1) * self.weight + self.bias


class TransformerAE(PyTorchModel):
    """
    Transformer-based autoencoder that treats each input feature as a token
    and uses self-attention to model cross-feature interactions.
    Reconstruction-based; same threshold logic as the deep AE.
    """

    def __init__(self, dataset_name, input_size, device,
                 d_model=32, nhead=4, num_layers=2,
                 dim_feedforward=64, dropout=0.0,
                 lr=1e-3, weight_decay=1e-5,
                 threshold_percentile=95.0):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.threshold_percentile = threshold_percentile

        super().__init__(dataset_name, input_size, device)

        self.tokenizer = FeatureTokenizer(input_size, d_model).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        ).to(self.device)
        self.head = nn.Linear(d_model, 1).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def get_model(self):
        return nn.Identity()

    def forward(self, x):
        x = x.to(self.device)
        tokens = self.tokenizer(x)             # [B, n_features, d_model]
        encoded = self.transformer(tokens)     # [B, n_features, d_model]
        out = self.head(encoded).squeeze(-1)   # [B, n_features]
        return out

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
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / batch_count if batch_count else float("inf")
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
            self.scheduler.step(avg_loss)

        self.calculate_threshold(train_loader)

    def calculate_threshold(self, train_loader):
        print("Calculating threshold on training data...")
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
        print("Running TransformerAE evaluation...")
        y_test, y_pred, reconstruction_errors = self.infer(test_loader)
        y_test = np.array(y_test).ravel()
        y_pred = np.array(y_pred).ravel()
        print(f"Evaluated {len(y_test)} samples | threshold: {self.threshold:.6f}")
        self.plot_anomaly(reconstruction_errors)
        super().evaluate(y_test, y_pred, reconstruction_errors)
