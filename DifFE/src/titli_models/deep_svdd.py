import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ._compat import get_pytorch_model_base

PyTorchModel = get_pytorch_model_base()


class DeepSVDD(PyTorchModel):
    """
    Deep SVDD (one-class) with autoencoder pretraining.

    Training is two-phase, following Ruff et al. 2018:

      Phase 1 (AE pretraining): Train encoder + temporary decoder as a vanilla
      MSE autoencoder for `pretrain_epochs`. This forces the encoder to learn
      meaningful, information-preserving representations.

      Phase 2 (SVDD): Discard the decoder, initialize the center c from the
      mean of the pretrained encoder's outputs, then minimize ||f(x) - c||^2
      with a smaller learning rate so the encoder refines (rather than
      collapses) around its pretrained representations.

    Anomaly score = ||f(x) - c||^2.
    """

    def __init__(self, dataset_name, input_size, device,
                 latent_dim=16, hidden_1=64, hidden_2=32,
                 pretrain_epochs=10,
                 lr_pretrain=1e-3, lr_svdd=1e-4,
                 weight_decay=1e-6,
                 threshold_percentile=95.0):
        self.latent_dim = latent_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.pretrain_epochs = pretrain_epochs
        self.lr_pretrain = lr_pretrain
        self.lr_svdd = lr_svdd
        self.weight_decay = weight_decay
        self.threshold_percentile = threshold_percentile

        super().__init__(dataset_name, input_size, device)

        self.encoder = self._build_encoder().to(self.device)
        self.center = None  # set during training (after pretraining)

        # Optimizer is recreated in train_model (different LR per phase).
        # For the load-then-eval path, this is unused.
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr_svdd, weight_decay=self.weight_decay
        )

    def _build_encoder(self):
        # Bias=False and affine=False BatchNorm: required by Deep SVDD to avoid
        # the trivial c=0 collapse (any bias term lets the network arbitrarily
        # shift outputs to match c).
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_1, bias=False),
            nn.BatchNorm1d(self.hidden_1, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_1, self.hidden_2, bias=False),
            nn.BatchNorm1d(self.hidden_2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_2, self.latent_dim, bias=False),
        )

    def _build_decoder(self):
        # Decoder is only used during AE pretraining, so it's allowed to have
        # bias terms. It's discarded before SVDD training begins.
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_2),
            nn.BatchNorm1d(self.hidden_2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_2, self.hidden_1),
            nn.BatchNorm1d(self.hidden_1, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_1, self.input_size),
        )

    def get_model(self):
        # Required by parent. Real layers are built in __init__ after attributes are set.
        return nn.Identity()

    def forward(self, x):
        return self.encoder(x.to(self.device))

    def _pretrain(self, train_loader, epochs):
        print(f"\n=== Phase 1/2: Pretraining encoder as AE for {epochs} epochs ===")
        decoder = self._build_decoder().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(decoder.parameters()),
            lr=self.lr_pretrain, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        for epoch in range(epochs):
            self.train()
            decoder.train()
            running_loss, batch_count = 0.0, 0
            for inputs, _ in tqdm(train_loader, desc=f"Pretrain Epoch {epoch + 1}"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)

                optimizer.zero_grad()
                z = self.encoder(inputs_scaled)
                recon = decoder(z)
                loss = criterion(recon, inputs_scaled)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / batch_count if batch_count else float("inf")
            print(f"Pretrain Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
            scheduler.step(avg_loss)

        # Decoder goes out of scope when this method returns; only the
        # pretrained encoder weights are retained.

    def _init_center(self, train_loader, eps=0.1):
        n, c = 0, torch.zeros(self.latent_dim, device=self.device)
        self.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Initializing SVDD center"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                z = self.encoder(inputs_scaled)
                n += z.shape[0]
                c += z.sum(dim=0)
        c /= n
        # Avoid components too close to zero (per the Deep SVDD paper)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.center = c
        print(f"Initialized SVDD center | norm={c.norm().item():.4f}")

    def train_model(self, train_loader, epochs=None):
        if epochs is None:
            epochs = self.epochs

        # Fit scaler once on the full training set
        all_train_data = [inputs.numpy() for inputs, _ in train_loader]
        all_train_data = np.concatenate(all_train_data, axis=0)
        self.scaler.fit(all_train_data)

        # --- Phase 1: AE pretraining ---
        if self.pretrain_epochs > 0:
            self._pretrain(train_loader, self.pretrain_epochs)

        # --- Phase 2: SVDD ---
        print(f"\n=== Phase 2/2: SVDD training for {epochs} epochs ===")
        self._init_center(train_loader)

        # Recreate optimizer with smaller LR — the encoder is already in a
        # good place, we just want it to refine, not collapse.
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr_svdd, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        for epoch in range(epochs):
            self.train()
            running_loss, batch_count = 0.0, 0
            for inputs, _ in tqdm(train_loader, desc=f"SVDD Epoch {epoch + 1}"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)

                self.optimizer.zero_grad()
                z = self(inputs_scaled)
                dist = ((z - self.center) ** 2).sum(dim=1)
                loss = dist.mean()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / batch_count if batch_count else float("inf")
            print(f"SVDD Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
            scheduler.step(avg_loss)

        self.calculate_threshold(train_loader)

    def calculate_threshold(self, train_loader):
        print("Calculating threshold on training data...")
        self.eval()
        per_sample_scores = []
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Calculating threshold"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                z = self(inputs_scaled)
                dist = ((z - self.center) ** 2).sum(dim=1).cpu().numpy()
                per_sample_scores.extend(dist)

        per_sample_scores = np.array(per_sample_scores)
        self.threshold = float(np.percentile(per_sample_scores, self.threshold_percentile))
        qs = np.percentile(per_sample_scores, [50, 75, 90, 95, 99, 99.5, 99.9])
        print(
            f"Threshold (p{self.threshold_percentile}): {self.threshold:.6f}\n"
            f"  mean: {per_sample_scores.mean():.6f} | "
            f"std: {per_sample_scores.std():.6f} | "
            f"max: {per_sample_scores.max():.6f}\n"
            f"  p50={qs[0]:.4f} p75={qs[1]:.4f} p90={qs[2]:.4f} "
            f"p95={qs[3]:.4f} p99={qs[4]:.4f} p99.5={qs[5]:.4f} p99.9={qs[6]:.4f}"
        )

    def infer(self, test_loader):
        self.eval()
        y_test, y_pred, scores = [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Inferencing"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                z = self(inputs_scaled)
                dist = ((z - self.center) ** 2).sum(dim=1).cpu().numpy()
                scores.extend(dist)
                y_test.extend(labels.cpu().numpy())
                y_pred.extend((dist > self.threshold).astype(int))
        return np.array(y_test), np.array(y_pred), scores

    def evaluate(self, test_loader):
        print("Running Deep SVDD evaluation...")
        y_test, y_pred, scores = self.infer(test_loader)
        y_test = y_test.ravel()
        y_pred = y_pred.ravel()
        print(f"Evaluated {len(y_test)} samples | threshold: {self.threshold:.6f}")
        self.plot_anomaly(scores)
        super().evaluate(y_test, y_pred, scores)

    def save(self, model_path=None):
        if not model_path:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pth"
        torch.save({
            "model_state_dict": self.state_dict(),
            "threshold": self.threshold,
            "scaler": self.scaler,
            "center": self.center.cpu() if self.center is not None else None,
        }, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path=None):
        if not model_path:
            model_path = f"./artifacts/{self.dataset_name}/models/{self.model_name.lower()}.pth"
        checkpoint = torch.load(model_path, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.threshold = checkpoint["threshold"]
        self.scaler = checkpoint["scaler"]
        if checkpoint.get("center") is not None:
            self.center = checkpoint["center"].to(self.device)
        return checkpoint
