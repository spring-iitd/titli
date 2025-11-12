Intrusion Detection Systems (titli.ids)
=======================================

The ``titli.ids`` module provides various intrusion detection system (IDS) models for anomaly detection in network traffic.

.. currentmodule:: titli.ids

Public API Overview
-------------------

All IDS models expose the following 5 public methods:

.. code-block:: python

   def train_model(self, train_loader: DataLoader) -> None
       """Train the model on training data."""
   
   def save(self, model_path: Optional[str] = None) -> None
       """Save trained model to disk.
       
       Args:
           model_path: Path to save model. If None, uses default path:
                      ./artifacts/{dataset_name}/models/{model_name}.pth
       """
   
   def load(self, model_path: Optional[str] = None) -> dict
       """Load trained model from disk.
       
       Args:
           model_path: Path to load model from. If None, uses default path.
       
       Returns:
           Checkpoint dictionary with model state
       """
   
   def infer(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
       """Lightweight inference without metrics computation.
       
       Args:
           test_loader: PyTorch DataLoader with test data
       
       Returns:
           Tuple of (y_true, y_pred, reconstruction_errors):
               - y_true: Ground truth labels
               - y_pred: Binary predictions (0=benign, 1=anomaly)
               - reconstruction_errors: Anomaly scores for each sample
       """
   
   def evaluate(self, test_loader: DataLoader) -> None
       """Full evaluation with metrics and visualization.
       
       Computes F1, Precision, Recall, Accuracy, AUC-ROC and generates:
       - Confusion matrix plot
       - ROC curve plot
       - Anomaly score plot
       - Metrics text file
       
       All artifacts saved to ./artifacts/{dataset_name}/
       
       Args:
           test_loader: PyTorch DataLoader with test data
       """

Available Models
----------------

Traditional ML Models
~~~~~~~~~~~~~~~~~~~~~

LOF (Local Outlier Factor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LOF
   :members: train_model, save, load, infer, evaluate
   :show-inheritance:
   :special-members: __init__

OCSVM (One-Class SVM)
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OCSVM
   :members: train_model, save, load, infer, evaluate
   :show-inheritance:
   :special-members: __init__

Deep Learning Models
~~~~~~~~~~~~~~~~~~~~

Autoencoder
^^^^^^^^^^^

.. autoclass:: Autoencoder
   :members: train_model, save, load, infer, evaluate
   :show-inheritance:
   :special-members: __init__

VAE (Variational Autoencoder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: VAE
   :members: train_model, save, load, infer, evaluate
   :show-inheritance:
   :special-members: __init__

ICL (Instance Contrastive Learning)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ICL
   :members: train_model, save, load, infer, evaluate
   :show-inheritance:
   :special-members: __init__

Ensemble Models
~~~~~~~~~~~~~~~

KitNET
^^^^^^

.. autoclass:: KitNET
   :members: train_model, save, load, infer, evaluate
   :show-inheritance:
   :special-members: __init__

Notes
-----

- All models inherit from either ``BaseSKLearnModel`` or ``PyTorchModel`` (internal base classes)
- Methods prefixed with ``_`` are internal and not part of the public API
- Default save paths: ``./artifacts/{dataset_name}/models/{model_name}.pth``
- ``infer()`` returns: ``(y_true, y_pred, reconstruction_errors)`` as numpy arrays
- ``evaluate()`` generates plots and metrics files in ``./artifacts/``
