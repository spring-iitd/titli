Quick Start
===========

This guide will help you get started with Titli quickly using the unified API.

Complete Workflow Example
--------------------------

Here's a complete example showing the standard 5-step workflow using OCSVM:

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

   # Step 5: Inference and evaluation
   test_loader = DataLoader(test_dataset, batch_size=32)
   y_true, y_pred, scores = ids.infer(test_loader)  # Lightweight
   ids.evaluate(test_loader)  # Full evaluation with plots

Understanding the Public API
-----------------------------

All 6 IDS models in Titli follow the same consistent API with exactly 5 public methods:

**1. train_model(train_loader)**
   Train the model on training data.
   
   .. code-block:: python
   
      ids.train_model(train_loader)

**2. save(model_path=None)**
   Save the trained model to disk.
   
   .. code-block:: python
   
      ids.save()  # Uses default path: ./artifacts/{dataset_name}/models/{model_name}.pth
      ids.save("custom_path.pth")  # Or specify custom path

**3. load(model_path=None)**
   Load a trained model from disk.
   
   .. code-block:: python
   
      ids.load()  # Uses default path
      ids.load("custom_path.pth")  # Or load from custom path

**4. infer(test_loader)**
   Lightweight inference returning predictions without computing metrics.
   
   .. code-block:: python
   
      y_true, y_pred, reconstruction_errors = ids.infer(test_loader)
      # Returns:
      #   y_true: Ground truth labels
      #   y_pred: Binary predictions (0=benign, 1=anomaly)
      #   reconstruction_errors: Anomaly scores

**5. evaluate(test_loader)**
   Full evaluation with metrics computation and visualization.
   
   .. code-block:: python
   
      ids.evaluate(test_loader)
      # Generates:
      #   - Confusion matrix plot
      #   - ROC curve plot
      #   - Anomaly score plot
      #   - Metrics file with F1, Precision, Recall, Accuracy, AUC

Switching Between Models
-------------------------

The unified API makes it easy to try different models with the same code pattern:

.. code-block:: python

   from titli.ids import LOF, OCSVM, VAE, Autoencoder, ICL, KitNET
   
   # All models use the same interface
   models = [
       LOF(dataset_name="test", input_size=100, device=device),
       OCSVM(dataset_name="test", input_size=100, device=device),
       Autoencoder(dataset_name="test", input_size=100, device=device),
       VAE(dataset_name="test", input_size=100, device=device),
       ICL(dataset_name="test", input_size=100, device=device),
       KitNET(dataset_name="test", input_size=100, device=device)
   ]
   
   for model in models:
       model.train_model(train_loader)
       model.save()
       model.evaluate(test_loader)

**Available Models:**

* **LOF** (Local Outlier Factor) - Traditional ML, density-based anomaly detection
* **OCSVM** (One-Class SVM) - Traditional ML, boundary-based anomaly detection
* **Autoencoder** - Deep learning, reconstruction-based anomaly detection
* **VAE** (Variational Autoencoder) - Deep learning with probabilistic latent space
* **ICL** (Instance Contrastive Learning) - Contrastive learning approach
* **KitNET** - Ensemble of autoencoders for online anomaly detection

Working with DataLoaders
-------------------------

Titli uses PyTorch DataLoaders for efficient data handling:

StreamingCSVDataset Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   
   # Create dataset from extracted features
   dataset = StreamingCSVDataset(
       feature_csv_path="features.csv",
       label_csv_path="labels.csv",
       max_samples=100000,
       label_column=0  # Column index for labels
   )
   
   # Create DataLoader with batching
   train_loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=False,
       num_workers=2
   )

Batch Processing Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large datasets, use appropriate batch sizes
   train_loader = DataLoader(dataset, batch_size=32)  # Standard
   test_loader = DataLoader(test_dataset, batch_size=64)  # Can be larger for inference
   
   # Train with batches
   ids.train_model(train_loader)
   
   # Infer on test data in batches
   y_true, y_pred, scores = ids.infer(test_loader)

Next Steps
----------

* Read the :doc:`usage` guide for detailed examples of each model
* Explore the :doc:`api/fe` documentation for feature extractors
* Check the :doc:`api/ids` documentation for detailed API reference
* See :doc:`api_reference` for the complete API contract
* View the example scripts in the ``examples/`` directory of the repository
