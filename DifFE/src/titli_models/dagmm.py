import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ._compat import get_pytorch_model_base

PyTorchModel = get_pytorch_model_base()


class DAGMM(PyTorchModel):
    """
    Deep Autoencoding Gaussian Mixture Model (Zong et al. 2018), with
    autoencoder pretraining and Cholesky-based stable energy computation.

    Training is two-phase:
      Phase 1 (AE pretrain): compression net trained as a vanilla MSE AE.
      Phase 2 (DAGMM): joint optimization of recon + sample_energy + cov_diag.

    Anomaly score = sample energy E(z) of the augmented latent z.
    """

    def __init__(self, dataset_name, input_size, device,
                 latent_dim=4, hidden_1=64, hidden_2=32,
                 n_gmm=4, lr=1e-4, weight_decay=1e-6,
                 lambda_energy=0.1, lambda_cov_diag=0.005,
                 pretrain_epochs=10, lr_pretrain=1e-3,
                 cov_eps=1e-6,
                 threshold_percentile=95.0):
        self.latent_dim = latent_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.n_gmm = n_gmm
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.pretrain_epochs = pretrain_epochs
        self.lr_pretrain = lr_pretrain
        self.cov_eps = cov_eps
        self.threshold_percentile = threshold_percentile

        super().__init__(dataset_name, input_size, device)

        # Built after attributes set
        self.encoder = self._build_encoder().to(self.device)
        self.decoder = self._build_decoder().to(self.device)
        self.estimation = self._build_estimation().to(self.device)

        gmm_dim = self.latent_dim + 2  # z_c + euclid + cosine
        self.register_buffer("phi", torch.zeros(self.n_gmm))
        self.register_buffer("mu", torch.zeros(self.n_gmm, gmm_dim))
        self.register_buffer("cov", torch.zeros(self.n_gmm, gmm_dim, gmm_dim))

        # Optimizer is recreated in train_model (different LR per phase).
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def get_model(self):
        return nn.Identity()

    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_1),
            nn.Tanh(),
            nn.Linear(self.hidden_1, self.hidden_2),
            nn.Tanh(),
            nn.Linear(self.hidden_2, self.latent_dim),
        )

    def _build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_2),
            nn.Tanh(),
            nn.Linear(self.hidden_2, self.hidden_1),
            nn.Tanh(),
            nn.Linear(self.hidden_1, self.input_size),
        )

    def _build_estimation(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim + 2, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, self.n_gmm),
            nn.Softmax(dim=1),
        )

    @staticmethod
    def _relative_euclid(x, recon, eps=1e-12):
        return ((x - recon).pow(2).sum(dim=1).sqrt() /
                (x.pow(2).sum(dim=1).sqrt() + eps)).unsqueeze(-1)

    @staticmethod
    def _cosine(x, recon, eps=1e-12):
        return F.cosine_similarity(x, recon, dim=1, eps=eps).unsqueeze(-1)

    def forward(self, x):
        x = x.to(self.device)
        z_c = self.encoder(x)
        recon = self.decoder(z_c)
        rec_feat = torch.cat(
            [self._relative_euclid(x, recon), self._cosine(x, recon)], dim=1
        )
        z = torch.cat([z_c, rec_feat], dim=1)
        gamma = self.estimation(z)
        return recon, z, gamma

    def _compute_gmm_params(self, z, gamma, eps=1e-12):
        # gamma: [N, K]; z: [N, D]
        N = gamma.size(0)
        sum_gamma = gamma.sum(dim=0).clamp(min=eps)   # [K]
        phi = sum_gamma / N                            # [K]

        mu = (gamma.unsqueeze(-1) * z.unsqueeze(1)).sum(dim=0) / sum_gamma.unsqueeze(-1)
        # mu: [K, D]

        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)        # [N, K, D]
        cov = (gamma.unsqueeze(-1).unsqueeze(-1) *
               z_mu.unsqueeze(-1) @ z_mu.unsqueeze(-2)).sum(dim=0)
        cov = cov / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        return phi, mu, cov

    def _sample_energy(self, z, phi, mu, cov):
        """
        Numerically stable energy via Cholesky decomposition + logsumexp.
        Returns per-sample energy (higher = more anomalous) and a cov_diag
        regularization term.
        """
        K, D = mu.size(0), mu.size(1)
        device = z.device
        eps = self.cov_eps

        log_two_pi = float(D) * float(np.log(2.0 * np.pi))

        log_components = []
        cov_diag_terms = []

        for k in range(K):
            cov_k = cov[k] + torch.eye(D, device=device) * eps
            try:
                L = torch.linalg.cholesky(cov_k)
            except Exception:
                # Fallback: stronger diagonal jitter
                cov_k = cov[k] + torch.eye(D, device=device) * (eps * 1e3)
                L = torch.linalg.cholesky(cov_k)

            # log det = 2 * sum(log(diag(L)))
            log_det = 2.0 * torch.log(torch.diagonal(L).clamp(min=1e-12)).sum()

            # Mahalanobis: z_mu_k^T (L L^T)^{-1} z_mu_k via cholesky_solve
            z_mu_k = (z - mu[k]).unsqueeze(-1)           # [N, D, 1]
            sol = torch.cholesky_solve(z_mu_k, L)        # [N, D, 1]
            mahal_sq = ((z - mu[k]) * sol.squeeze(-1)).sum(dim=1)  # [N]

            log_phi = torch.log(phi[k].clamp(min=1e-12))
            log_p_k = log_phi - 0.5 * (log_two_pi + log_det + mahal_sq)
            log_components.append(log_p_k)

            # cov_diag penalty
            cov_diag_terms.append(
                (1.0 / cov_k.diagonal().clamp(min=eps)).sum()
            )

        log_p = torch.stack(log_components, dim=1)       # [N, K]
        log_sum = torch.logsumexp(log_p, dim=1)
        sample_energy = -log_sum
        cov_diag = torch.stack(cov_diag_terms).sum()

        return sample_energy, cov_diag

    def _pretrain(self, train_loader, epochs):
        print(f"\n=== Phase 1/2: Pretraining compression net as AE for {epochs} epochs ===")
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr_pretrain, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        for epoch in range(epochs):
            self.train()
            running_loss, batch_count = 0.0, 0
            for inputs, _ in tqdm(train_loader, desc=f"Pretrain Epoch {epoch + 1}"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                optimizer.zero_grad()
                z_c = self.encoder(inputs_scaled)
                recon = self.decoder(z_c)
                loss = criterion(recon, inputs_scaled)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
            avg_loss = running_loss / batch_count if batch_count else float("inf")
            print(f"Pretrain Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
            scheduler.step(avg_loss)

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

        # --- Phase 2: DAGMM joint training ---
        print(f"\n=== Phase 2/2: DAGMM training for {epochs} epochs ===")
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        for epoch in range(epochs):
            self.train()
            running_loss, batch_count = 0.0, 0
            skipped = 0
            for inputs, _ in tqdm(train_loader, desc=f"DAGMM Epoch {epoch + 1}"):
                if inputs.shape[0] < 2 * self.n_gmm:
                    # need enough samples to populate every mixture
                    continue
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)

                self.optimizer.zero_grad()
                recon, z, gamma = self(inputs_scaled)
                recon_loss = F.mse_loss(recon, inputs_scaled)

                phi, mu, cov = self._compute_gmm_params(z, gamma)
                sample_energy, cov_diag = self._sample_energy(z, phi, mu, cov)

                energy_term = sample_energy.mean()
                loss = (recon_loss
                        + self.lambda_energy * energy_term
                        + self.lambda_cov_diag * cov_diag)

                if not torch.isfinite(loss):
                    skipped += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / batch_count if batch_count else float("inf")
            print(f"DAGMM Epoch {epoch + 1}, Loss: {avg_loss:.6f} (skipped {skipped} batches)")
            scheduler.step(avg_loss)

        self._fit_gmm(train_loader)
        self.calculate_threshold(train_loader)

    def _fit_gmm(self, train_loader):
        """Compute final phi/mu/cov over the full training set."""
        self.eval()
        zs, gammas = [], []
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Fitting final GMM"):
                inputs_scaled = self.scaler.transform(inputs)
                inputs_scaled = torch.tensor(
                    inputs_scaled, dtype=torch.float32
                ).to(self.device)
                _, z, gamma = self(inputs_scaled)
                zs.append(z)
                gammas.append(gamma)
        z_all = torch.cat(zs, dim=0)
        gamma_all = torch.cat(gammas, dim=0)
        phi, mu, cov = self._compute_gmm_params(z_all, gamma_all)
        self.phi.copy_(phi.detach())
        self.mu.copy_(mu.detach())
        self.cov.copy_(cov.detach())

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
                _, z, _ = self(inputs_scaled)
                energy, _ = self._sample_energy(z, self.phi, self.mu, self.cov)
                per_sample_scores.extend(energy.cpu().numpy())

        per_sample_scores = np.array(per_sample_scores, dtype=np.float64)
        finite_mask = np.isfinite(per_sample_scores)
        if not finite_mask.any():
            raise RuntimeError(
                "All training energies are non-finite; DAGMM training failed. "
                "Check that pretraining converged and consider lowering "
                "lambda_energy or raising cov_eps."
            )
        finite_scores = per_sample_scores[finite_mask]
        # Replace any NaN/Inf with median / max-finite / min-finite
        per_sample_scores = np.where(
            finite_mask, per_sample_scores, np.median(finite_scores)
        )

        self.threshold = float(np.percentile(per_sample_scores, self.threshold_percentile))
        qs = np.percentile(per_sample_scores, [50, 75, 90, 95, 99, 99.5, 99.9])
        print(
            f"Threshold (p{self.threshold_percentile}): {self.threshold:.6f}\n"
            f"  finite: {finite_mask.sum()}/{len(per_sample_scores)} | "
            f"mean: {per_sample_scores.mean():.4f} | "
            f"std: {per_sample_scores.std():.4f}\n"
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
                _, z, _ = self(inputs_scaled)
                energy, _ = self._sample_energy(z, self.phi, self.mu, self.cov)
                e = energy.cpu().numpy()
                e = np.nan_to_num(
                    e,
                    nan=self.threshold,
                    posinf=self.threshold * 10,
                    neginf=0.0,
                )
                scores.extend(e)
                y_test.extend(labels.cpu().numpy())
                y_pred.extend((e > self.threshold).astype(int))
        return np.array(y_test), np.array(y_pred), scores

    def evaluate(self, test_loader):
        print("Running DAGMM evaluation...")
        y_test, y_pred, scores = self.infer(test_loader)
        y_test = y_test.ravel()
        y_pred = y_pred.ravel()
        print(f"Evaluated {len(y_test)} samples | threshold: {self.threshold:.6f}")
        self.plot_anomaly(scores)
        super().evaluate(y_test, y_pred, scores)
