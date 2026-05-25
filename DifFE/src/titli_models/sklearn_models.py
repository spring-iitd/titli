import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest as SkIsolationForest

from ._compat import get_create_directories, get_pytorch_model_base

PyTorchModel = get_pytorch_model_base()
create_directories = get_create_directories()


class _SklearnBase:
    """
    Lightweight base for sklearn-based anomaly models. Mimics the PyTorchModel
    interface (model_name, dataset_name, scaler, threshold, save/load,
    evaluate) so it produces identical artifact paths and metric files.
    """

    def __init__(self, dataset_name, input_size, device,
                 threshold_percentile=95.0):
        self.model_name = self.__class__.__name__
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.device = device  # ignored
        self.threshold_percentile = threshold_percentile

        self.scaler = StandardScaler()
        self.threshold = None
        self.model = None  # populated by subclass

        create_directories(dataset_name)

    # --- subclasses override these ---
    def _build_model(self):
        raise NotImplementedError

    def _score(self, X_scaled):
        """Return per-sample anomaly score (higher = more anomalous)."""
        raise NotImplementedError

    def _fit(self, X_scaled):
        self.model.fit(X_scaled)

    # --- common pipeline ---
    def _collect(self, loader):
        X, y = [], []
        for inputs, labels in loader:
            X.append(inputs.numpy())
            y.append(labels.numpy())
        return np.vstack(X), np.hstack(y)

    def train_model(self, train_loader, epochs=None):
        # epochs ignored
        print(f"Collecting training data for {self.model_name}...")
        X, _ = self._collect(train_loader)
        X_scaled = self.scaler.fit_transform(X)
        print(f"Fitting {self.model_name} on {X_scaled.shape[0]} samples...")
        self._fit(X_scaled)
        self.calculate_threshold(X_scaled)

    def calculate_threshold(self, X_scaled):
        print("Calculating threshold on training scores...")
        scores = self._score(X_scaled)
        self.threshold = float(np.percentile(scores, self.threshold_percentile))
        qs = np.percentile(scores, [50, 75, 90, 95, 99, 99.5, 99.9])
        print(
            f"Threshold (p{self.threshold_percentile}): {self.threshold:.6f}\n"
            f"  mean: {scores.mean():.6f} | std: {scores.std():.6f} | "
            f"max: {scores.max():.6f}\n"
            f"  p50={qs[0]:.4f} p75={qs[1]:.4f} p90={qs[2]:.4f} "
            f"p95={qs[3]:.4f} p99={qs[4]:.4f} p99.5={qs[5]:.4f} p99.9={qs[6]:.4f}"
        )

    def infer(self, test_loader):
        X, y_test = self._collect(test_loader)
        X_scaled = self.scaler.transform(X)
        scores = self._score(X_scaled)
        y_pred = (scores > self.threshold).astype(int)
        return y_test, y_pred, list(scores)

    def evaluate(self, test_loader):
        print(f"Running {self.model_name} evaluation...")
        y_test, y_pred, scores = self.infer(test_loader)
        y_test = np.array(y_test).ravel()
        y_pred = np.array(y_pred).ravel()
        print(f"Evaluated {len(y_test)} samples | threshold: {self.threshold:.6f}")
        # Reuse PyTorchModel.plot_anomaly and PyTorchModel.evaluate by binding self.
        PyTorchModel.plot_anomaly(self, scores)
        PyTorchModel.evaluate(self, y_test, y_pred, scores)

    def save(self, model_path=None):
        if not model_path:
            model_path = (
                f"./artifacts/{self.dataset_name}/models/"
                f"{self.model_name.lower()}.pkl"
            )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "threshold": self.threshold,
            }, f)
        print(f"Model saved to {model_path}")

    def load(self, model_path=None):
        if not model_path:
            model_path = (
                f"./artifacts/{self.dataset_name}/models/"
                f"{self.model_name.lower()}.pkl"
            )
        with open(model_path, "rb") as f:
            ckpt = pickle.load(f)
        self.model = ckpt["model"]
        self.scaler = ckpt["scaler"]
        self.threshold = ckpt["threshold"]
        print(f"Model loaded from {model_path}")


class LOF(_SklearnBase):
    """Local Outlier Factor (novelty=True for unseen-data scoring)."""

    def __init__(self, dataset_name, input_size, device,
                 n_neighbors=20, contamination="auto",
                 threshold_percentile=95.0, n_jobs=-1):
        super().__init__(dataset_name, input_size, device, threshold_percentile)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=n_jobs,
        )

    def _score(self, X_scaled):
        # score_samples is higher for inliers, so negate.
        return -self.model.score_samples(X_scaled)


class OCSVM(_SklearnBase):
    """One-Class SVM with RBF kernel."""

    def __init__(self, dataset_name, input_size, device,
                 nu=0.05, gamma="scale", kernel="rbf",
                 threshold_percentile=95.0):
        super().__init__(dataset_name, input_size, device, threshold_percentile)
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.model = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)

    def _score(self, X_scaled):
        # decision_function: higher = more inlier
        return -self.model.decision_function(X_scaled)


class IsolationForest(_SklearnBase):
    """Isolation Forest."""

    def __init__(self, dataset_name, input_size, device,
                 n_estimators=500, contamination="auto",
                 max_samples="auto", max_features=1.0,
                 bootstrap=False, random_state=42, n_jobs=-1,
                 threshold_percentile=80.0):
        super().__init__(dataset_name, input_size, device, threshold_percentile)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = SkIsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _score(self, X_scaled):
        # score_samples: higher = more inlier
        return -self.model.score_samples(X_scaled)
