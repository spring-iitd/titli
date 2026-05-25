import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ._compat import get_pytorch_model_base

PyTorchModel = get_pytorch_model_base()


class VAE(PyTorchModel):
    """
    Drop-in replacement for titli.ids.VAE with a deeper architecture,
    beta-weighted KL divergence, and per-sample threshold calibration.
    """

    def __init__(self, dataset_name, input_size, device,
                 latent_dim=16, hidden_1=64, hidden_2=32,
                 lr=1e-3, weight_decay=1e-5, beta=0.1,
                 threshold_percentile=95.0):
        self.latent_dim = latent_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.beta = beta
        self.threshold_percentile = threshold_percentile

        super().__init__(dataset_name, input_size, device)

        self.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def get_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_1),
            nn.BatchNorm1d(self.hidden_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_1, self.hidden_2),
            nn.BatchNorm1d(self.hidden_2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(self.hidden_2, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_2, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_2),
            nn.BatchNorm1d(self.hidden_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_2, self.hidden_1),
            nn.BatchNorm1d(self.hidden_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_1, self.input_size),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.to(self.device)
        mu, logvar = self.encode(x)
        logvar = torch.clip(logvar, max=10.0)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld, recon_loss, kld

    def reconstruction_error(self, recon_x, x):
        return F.mse_loss(recon_x, x, reduction="none").mean(dim=1)

    def train_model(self, train_loader, epochs=None):
        if epochs is None:
            epochs = self.epochs

        all_train_data = [inputs.numpy() for inputs, _ in train_loader]
        all_train_data = np.concatenate(all_train_data, axis=0)
        self.scaler.fit(all_train_data)

        for epoch in range(epochs):
            self.train()
            running_loss, running_recon, running_kld, batch_count = 0.0, 0.0, 0.0, 0
            for inputs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)

                self.optimizer.zero_grad()
                recon, mu, logvar = self(inputs_scaled)
                loss, recon_loss, kld = self.loss_function(
                    recon, inputs_scaled, mu, logvar
                )
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                running_recon += recon_loss.item()
                running_kld += kld.item()
                batch_count += 1

            denom = batch_count if batch_count else 1
            avg_loss = running_loss / denom
            print(
                f"Epoch {epoch + 1}, Loss: {avg_loss:.6f} "
                f"(recon: {running_recon / denom:.6f}, kld: {running_kld / denom:.6f})"
            )
            self.scheduler.step(avg_loss)

        self.calculate_threshold(train_loader)

    def calculate_threshold(self, train_loader):
        print("Calculating per-sample threshold on training data...")
        self.eval()
        per_sample_errors = []
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Calculating threshold"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                recon, _, _ = self(inputs_scaled)
                errors = self.reconstruction_error(recon, inputs_scaled).cpu().numpy()
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

    def infer(self, test_loader):
        self.eval()
        y_true, y_pred, reconstruction_errors = [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Inferencing"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                recon, _, _ = self(inputs_scaled)
                errors = self.reconstruction_error(recon, inputs_scaled).cpu().numpy()
                reconstruction_errors.extend(errors)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend((errors > self.threshold).astype(int))
        return np.array(y_true), np.array(y_pred), reconstruction_errors

    def evaluate(self, test_loader):
        print("Running VAE evaluation...")
        y_test, y_pred, reconstruction_errors = self.infer(test_loader)
        y_test = np.array(y_test).ravel()
        y_pred = np.array(y_pred).ravel()

        print(f"Evaluated {len(y_test)} samples")
        print(f"Threshold: {self.threshold:.6f}")

        self.plot_anomaly(reconstruction_errors)
        super().evaluate(y_test, y_pred, reconstruction_errors)
