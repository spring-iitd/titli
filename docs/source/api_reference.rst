API Reference Summary
=====================

This document provides a consolidated view of the unified API that all IDS models implement.

Public API Contract
-------------------

All 6 IDS models (LOF, OCSVM, VAE, Autoencoder, ICL, KitNET) implement the same public API:

Method Signatures
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BaseIDSModel:
       """Base interface for all IDS models"""
       
       def train_model(self, train_loader: DataLoader) -> None:
           """Train the model on training data.
           
           Args:
               train_loader: PyTorch DataLoader with training data
           """
       
       def save(self, model_path: Optional[str] = None) -> None:
           """Save trained model to disk.
           
           Args:
               model_path: Path to save model. If None, uses default path:
                          ./artifacts/{dataset_name}/models/{model_name}.pth
           """
       
       def load(self, model_path: Optional[str] = None) -> dict:
           """Load trained model from disk.
           
           Args:
               model_path: Path to load model from. If None, uses default path.
           
           Returns:
               Checkpoint dictionary with model state
           """
       
       def infer(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
           """Lightweight inference without metrics computation.
           
           Args:
               test_loader: PyTorch DataLoader with test data
           
           Returns:
               Tuple of (y_true, y_pred, reconstruction_errors):
                   - y_true: Ground truth labels
                   - y_pred: Binary predictions (0=benign, 1=anomaly)
                   - reconstruction_errors: Anomaly scores for each sample
           """
       
       def evaluate(self, test_loader: DataLoader) -> None:
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

Usage Pattern
~~~~~~~~~~~~~

Every model follows this exact pattern:

.. code-block:: python

   from titli.ids import ModelName  # Any of: LOF, OCSVM, VAE, Autoencoder, ICL, KitNET
   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   import torch
   
   # Setup data
   dataset = StreamingCSVDataset(
       feature_csv_path="features.csv",
       label_csv_path="labels.csv"
   )
   train_loader = DataLoader(dataset, batch_size=32)
   
   # Train
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = ModelName(
       dataset_name="my_dataset",
       input_size=dataset.input_size,
       device=device
   )
   model.train_model(train_loader)
   
   # Persist
   model.save()  # or model.save("custom_path.pth")
   model.load()  # or model.load("custom_path.pth")
   
   # Inference
   y_true, y_pred, scores = model.infer(test_loader)
   
   # Evaluation
   model.evaluate(test_loader)

Model Descriptions
------------------

LOF (Local Outlier Factor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Type**: Traditional Machine Learning

**Description**: Density-based anomaly detection using local outlier factors.

**When to use**: Small to medium datasets with clear density-based outliers. Works well when 
anomalies have significantly different local densities than normal samples.

**Initialization**:

.. code-block:: python

   from titli.ids import LOF
   
   model = LOF(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

OCSVM (One-Class SVM)
~~~~~~~~~~~~~~~~~~~~~~

**Type**: Traditional Machine Learning

**Description**: Boundary-based anomaly detection using support vector machines.

**When to use**: Datasets with clear decision boundaries. Effective for high-dimensional data 
and when you want a well-defined separation between normal and anomalous regions.

**Initialization**:

.. code-block:: python

   from titli.ids import OCSVM
   
   model = OCSVM(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

Autoencoder
~~~~~~~~~~~

**Type**: Deep Learning

**Description**: Reconstruction-based anomaly detection using neural networks.

**When to use**: Complex patterns in high-dimensional data, GPU available. Learns to reconstruct 
normal patterns; anomalies produce higher reconstruction errors.

**Initialization**:

.. code-block:: python

   from titli.ids import Autoencoder
   
   model = Autoencoder(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

VAE (Variational Autoencoder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Type**: Deep Learning

**Description**: Probabilistic deep learning with latent space modeling.

**When to use**: When you need probabilistic anomaly scores or want to model the distribution 
of normal data in a latent space. Better for capturing uncertainty than standard autoencoders.

**Initialization**:

.. code-block:: python

   from titli.ids import VAE
   
   model = VAE(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

ICL (Instance Contrastive Learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Type**: Contrastive Learning

**Description**: Contrastive learning approach for anomaly detection.

**When to use**: When you want to learn discriminative features through contrastive learning. 
Effective for scenarios where normal samples should cluster together in feature space.

**Initialization**:

.. code-block:: python

   from titli.ids import ICL
   
   model = ICL(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

KitNET
~~~~~~

**Type**: Ensemble

**Description**: Ensemble of autoencoders for online anomaly detection.

**When to use**: Online/streaming scenarios, ensemble methods needed. KitNET adaptively creates 
an ensemble of small autoencoders, making it efficient for incremental learning.

**Initialization**:

.. code-block:: python

   from titli.ids import KitNET
   
   model = KitNET(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

Complete Workflow Example
--------------------------

Here's a complete example showing feature extraction through evaluation:

.. code-block:: python

   from titli.fe import AfterImage
   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   from titli.ids import OCSVM
   import torch
   
   # Step 1: Extract features
   fe = AfterImage(file_path="traffic.pcap")
   fe.extract_features(output_path="features.csv")
   
   # Step 2: Create DataLoader
   dataset = StreamingCSVDataset(
       feature_csv_path="features.csv",
       label_csv_path="labels.csv",
       max_samples=100000,
       label_column=0
   )
   train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
   
   # Step 3: Train model
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ids = OCSVM(dataset_name="my_dataset", input_size=dataset.input_size, device=device)
   ids.train_model(train_loader)
   
   # Step 4: Save and load
   ids.save()  # Saves to ./artifacts/my_dataset/models/ocsvm.pth
   ids.load()  # Loads from same path
   
   # Step 5: Inference
   test_loader = DataLoader(test_dataset, batch_size=32)
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Step 6: Full evaluation
   ids.evaluate(test_loader)

Output Artifacts
----------------

Default Paths
~~~~~~~~~~~~~

All models save artifacts to standardized locations:

.. code-block:: text

   ./artifacts/{dataset_name}/
   ├── models/
   │   └── {model_name}.pth          # Trained model
   ├── objects/
   │   └── metrics/
   │       └── {model_name}.txt      # Metrics file
   └── plots/
       ├── confusion_matrix/
       │   └── {model_name}.png      # Confusion matrix plot
       ├── roc/
       │   └── {model_name}.png      # ROC curve plot
       └── anomaly/
           └── {model_name}.png      # Anomaly score plot

Metrics File Content
~~~~~~~~~~~~~~~~~~~~

Example ``metrics.txt`` content:

.. code-block:: text

   Accuracy:    0.956
   Precision:   0.892
   Recall(TPR): 0.847
   F1 Score:    0.869
   
   Confusion Matrix:
   TP: 1234
   TN: 8765
   FP: 234
   FN: 167
   TPR (Recall): 0.847
   FNR:          0.153
   FPR:          0.026
   TNR:          0.974
   
   AUC-ROC:      0.9234

Key Differences: infer() vs evaluate()
--------------------------------------

**infer()** - Lightweight Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Purpose**: Get predictions without generating artifacts
* **Use cases**: 
    - Online detection systems
    - Streaming scenarios
    - Custom workflows
    - Integration with external systems
* **Returns**: ``(y_true, y_pred, reconstruction_errors)``
* **No side effects**: Doesn't save anything to disk

**evaluate()** - Full Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Purpose**: Comprehensive model evaluation with visualization
* **Use cases**:
    - Model benchmarking
    - Performance analysis
    - Development and experimentation
    - Generating reports
* **Returns**: ``None``
* **Side effects**: Creates plots and metrics files

Example Comparison
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use infer() for production/online detection
   y_true, y_pred, scores = model.infer(test_loader)
   # Now use predictions in your application
   anomaly_rate = y_pred.sum() / len(y_pred)
   send_alert_if_threshold_exceeded(anomaly_rate)
   
   # Use evaluate() for development/analysis
   model.evaluate(test_loader)
   # Generates all plots and metrics automatically
   # Review ./artifacts/{dataset_name}/ for results

Best Practices
--------------

1. **Always use DataLoaders**: Never pass raw arrays to train/infer/evaluate methods
2. **Consistent batch sizes**: Use smaller batches (16-32) for training, larger for inference (64+)
3. **Save frequently**: Call ``save()`` after training to preserve your work
4. **Use default paths**: Let Titli manage paths automatically unless you have specific needs
5. **Choose the right method**: Use ``infer()`` in production, ``evaluate()`` for development
6. **Try multiple models**: The unified API makes it trivial to compare different models
7. **GPU when available**: Pass ``device=torch.device("cuda")`` for deep learning models

Common Patterns
---------------

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   from titli.ids import LOF, OCSVM, Autoencoder
   
   models = {
       "LOF": LOF(dataset_name="comparison", input_size=100, device=device),
       "OCSVM": OCSVM(dataset_name="comparison", input_size=100, device=device),
       "Autoencoder": Autoencoder(dataset_name="comparison", input_size=100, device=device)
   }
   
   for name, model in models.items():
       print(f"Training {name}...")
       model.train_model(train_loader)
       model.save()
       model.evaluate(test_loader)
       print(f"{name} complete!\n")

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import KFold
   
   kfold = KFold(n_splits=5)
   
   for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
       # Create fold-specific loaders
       train_loader = create_loader(data[train_idx])
       val_loader = create_loader(data[val_idx])
       
       # Train and evaluate
       model = OCSVM(dataset_name=f"fold_{fold}", input_size=100, device=device)
       model.train_model(train_loader)
       model.evaluate(val_loader)

Hyperparameter Search
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   learning_rates = [0.001, 0.01, 0.1]
   
   for lr in learning_rates:
       model = Autoencoder(
           dataset_name=f"lr_{lr}",
           input_size=100,
           device=device
       )
       model.learning_rate = lr  # Set hyperparameter
       model.train_model(train_loader)
       model.evaluate(test_loader)
