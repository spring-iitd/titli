Usage Guide
===========

This guide provides detailed examples of using Titli for various IDS tasks.

Feature Extraction
------------------

AfterImage Feature Extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AfterImage is a packet-based feature extractor that extracts temporal and statistical features from network traffic.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from titli.fe import AfterImage
   
   # Extract features and output to CSV for DataLoader consumption
   fe = AfterImage(file_path="traffic.pcap")
   fe.extract_features(output_path="features.csv")

With Custom Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from titli.fe import AfterImage
   
   fe = AfterImage(
       file_path="path/to/traffic.pcap",
       limit=float("inf"),  # Maximum number of records
       decay_factors=[5, 3, 1, 0.1, 0.01],  # Time windows
       max_pkt=float("inf")  # Maximum packets to process
   )
   fe.extract_features(output_path="features.csv")

.. note::

   Feature extractors output CSV files that should be consumed via ``StreamingCSVDataset`` and ``DataLoader`` 
   for model training. State management is now handled internally by the feature extractors.

DataLoader Setup
----------------

StreamingCSVDataset
~~~~~~~~~~~~~~~~~~~

Create datasets from extracted features for efficient batch processing:

.. code-block:: python

   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   
   # Create dataset
   dataset = StreamingCSVDataset(
       feature_csv_path="features.csv",
       label_csv_path="labels.csv",
       max_samples=100000,
       label_column=0  # Column index containing labels
   )
   
   # Create DataLoader
   train_loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=False,
       num_workers=2
   )

Batch Size Selection
~~~~~~~~~~~~~~~~~~~~

Choose appropriate batch sizes based on your use case:

.. code-block:: python

   # Small batches for memory-constrained environments
   train_loader = DataLoader(dataset, batch_size=16)
   
   # Standard batches for most cases
   train_loader = DataLoader(dataset, batch_size=32)
   
   # Larger batches for inference (no gradients)
   test_loader = DataLoader(test_dataset, batch_size=64)

Train/Test Split Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch.utils.data import random_split
   
   # Split dataset
   train_size = int(0.8 * len(dataset))
   test_size = len(dataset) - train_size
   train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
   
   train_loader = DataLoader(train_dataset, batch_size=32)
   test_loader = DataLoader(test_dataset, batch_size=32)

Model Training
--------------

All models follow the same 5-method workflow. Below are examples for each of the 6 available models.

LOF (Local Outlier Factor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Density-based anomaly detection using local outlier factors.

**When to use**: Small to medium datasets with clear density-based outliers. Works well when anomalies 
have significantly different local densities than normal samples.

**Initialization**:

.. code-block:: python

   from titli.ids import LOF
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ids = LOF(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

**Complete workflow**:

.. code-block:: python

   # Train
   ids.train_model(train_loader)
   
   # Save
   ids.save()
   
   # Load
   ids.load()
   
   # Infer
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Evaluate
   ids.evaluate(test_loader)

OCSVM (One-Class SVM)
~~~~~~~~~~~~~~~~~~~~~~

Boundary-based anomaly detection using support vector machines.

**When to use**: Datasets with clear decision boundaries. Effective for high-dimensional data 
and when you want a well-defined separation between normal and anomalous regions.

**Initialization**:

.. code-block:: python

   from titli.ids import OCSVM
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ids = OCSVM(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

**Complete workflow**:

.. code-block:: python

   # Train
   ids.train_model(train_loader)
   
   # Save
   ids.save()
   
   # Load
   ids.load()
   
   # Infer
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Evaluate
   ids.evaluate(test_loader)

Autoencoder
~~~~~~~~~~~

Deep learning reconstruction-based anomaly detection.

**When to use**: Complex patterns in high-dimensional data, GPU available. Learns to reconstruct 
normal patterns; anomalies produce higher reconstruction errors.

**Initialization**:

.. code-block:: python

   from titli.ids import Autoencoder
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ids = Autoencoder(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

**Complete workflow**:

.. code-block:: python

   # Train
   ids.train_model(train_loader)
   
   # Save
   ids.save()
   
   # Load
   ids.load()
   
   # Infer
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Evaluate
   ids.evaluate(test_loader)

VAE (Variational Autoencoder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Probabilistic deep learning with latent space modeling.

**When to use**: When you need probabilistic anomaly scores or want to model the distribution 
of normal data in a latent space. Better for capturing uncertainty than standard autoencoders.

**Initialization**:

.. code-block:: python

   from titli.ids import VAE
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ids = VAE(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

**Complete workflow**:

.. code-block:: python

   # Train
   ids.train_model(train_loader)
   
   # Save
   ids.save()
   
   # Load
   ids.load()
   
   # Infer
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Evaluate
   ids.evaluate(test_loader)

ICL (Instance Contrastive Learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contrastive learning approach for anomaly detection.

**When to use**: When you want to learn discriminative features through contrastive learning. 
Effective for scenarios where normal samples should cluster together in feature space.

**Initialization**:

.. code-block:: python

   from titli.ids import ICL
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ids = ICL(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

**Complete workflow**:

.. code-block:: python

   # Train
   ids.train_model(train_loader)
   
   # Save
   ids.save()
   
   # Load
   ids.load()
   
   # Infer
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Evaluate
   ids.evaluate(test_loader)

KitNET
~~~~~~

Ensemble of autoencoders for online anomaly detection.

**When to use**: Online/streaming scenarios, ensemble methods needed. KitNET adaptively creates 
an ensemble of small autoencoders, making it efficient for incremental learning.

**Initialization**:

.. code-block:: python

   from titli.ids import KitNET
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ids = KitNET(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )

**Complete workflow**:

.. code-block:: python

   # Train
   ids.train_model(train_loader)
   
   # Save
   ids.save()
   
   # Load
   ids.load()
   
   # Infer
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Evaluate
   ids.evaluate(test_loader)

Model Persistence
-----------------

Save and Load Patterns
~~~~~~~~~~~~~~~~~~~~~~

All models use the same save/load interface:

.. code-block:: python

   # Save with default path
   ids.save()  # Saves to ./artifacts/{dataset_name}/models/{model_name}.pth
   
   # Save with custom path
   ids.save("custom_model_path.pth")
   
   # Load with default path
   ids.load()  # Loads from ./artifacts/{dataset_name}/models/{model_name}.pth
   
   # Load from custom path
   ids.load("custom_model_path.pth")

Default Paths
~~~~~~~~~~~~~

Models are saved to standardized locations:

.. code-block:: python

   # For OCSVM with dataset_name="traffic_analysis"
   # Default save path: ./artifacts/traffic_analysis/models/ocsvm.pth
   
   # For Autoencoder with dataset_name="network_ids"
   # Default save path: ./artifacts/network_ids/models/autoencoder.pth

Custom Paths
~~~~~~~~~~~~

.. code-block:: python

   # Save to custom location
   ids.save("/path/to/my_model.pth")
   
   # Later, load from that location
   ids.load("/path/to/my_model.pth")

Inference Patterns
------------------

When to Use infer() vs evaluate()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use infer()** for:

* Online detection and streaming scenarios
* Custom workflows where you need raw predictions
* Integration with external systems
* When you don't need visualization or metrics files

.. code-block:: python

   # Lightweight inference - just get predictions
   y_true, y_pred, scores = ids.infer(test_loader)
   
   # Now use predictions in custom workflow
   anomaly_rate = y_pred.sum() / len(y_pred)
   print(f"Detected {anomaly_rate:.2%} anomalies")

**Use evaluate()** for:

* Benchmarking and model comparison
* Generating reports with visualizations
* Computing comprehensive metrics
* Development and experimentation

.. code-block:: python

   # Full evaluation - generates all artifacts
   ids.evaluate(test_loader)
   # Creates:
   # - ./artifacts/{dataset_name}/plots/confusion_matrix/{model}.png
   # - ./artifacts/{dataset_name}/plots/roc/{model}.png
   # - ./artifacts/{dataset_name}/plots/anomaly/{model}.png
   # - ./artifacts/{dataset_name}/objects/metrics/{model}.txt

Output Artifacts
----------------

Metrics Files
~~~~~~~~~~~~~

After calling ``evaluate()``, metrics are saved to:

``./artifacts/{dataset_name}/objects/metrics/{model_name}.txt``

Example content:

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

Plots
~~~~~

Three types of plots are generated:

**1. Confusion Matrix**
   ``./artifacts/{dataset_name}/plots/confusion_matrix/{model_name}.png``
   
   Shows true positives, false positives, true negatives, and false negatives.

**2. ROC Curve**
   ``./artifacts/{dataset_name}/plots/roc/{model_name}.png``
   
   Shows the trade-off between true positive rate and false positive rate.

**3. Anomaly Score Plot**
   ``./artifacts/{dataset_name}/plots/anomaly/{model_name}.png``
   
   Shows anomaly scores for all samples with the threshold line.

File Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~

All output files follow consistent naming:

.. code-block:: text

   ./artifacts/
   └── {dataset_name}/
       ├── models/
       │   ├── lof.pth
       │   ├── ocsvm.pth
       │   ├── autoencoder.pth
       │   ├── vae.pth
       │   ├── icl.pth
       │   └── kitnet.pth
       ├── objects/
       │   └── metrics/
       │       ├── lof.txt
       │       ├── ocsvm.txt
       │       └── ...
       └── plots/
           ├── confusion_matrix/
           │   ├── lof.png
           │   └── ...
           ├── roc/
           │   ├── lof.png
           │   └── ...
           └── anomaly/
               ├── lof.png
               └── ...

Tips and Best Practices
------------------------

1. **DataLoader Usage**: Always use DataLoaders for efficient batch processing
2. **Batch Size**: Start with 32; increase for inference, decrease if memory constrained
3. **Model Selection**: Try multiple models - the unified API makes this easy
4. **Save Frequently**: Save models after training to avoid losing progress
5. **Use infer() for Production**: Use ``infer()`` in production; ``evaluate()`` for development
6. **GPU Acceleration**: Use GPU (``device="cuda"``) for deep learning models when available

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**PCAP File Not Found**
   Ensure the path to your PCAP file is correct and the file exists.

**Out of Memory**
   Reduce batch size or use ``num_workers=0`` in DataLoader.

**Model Not Converging**
   Deep learning models: Adjust learning rate or increase epochs.
   Traditional ML: Check data preprocessing and scaling.

**Poor Detection Performance**
   * Ensure proper feature normalization
   * Try different models - some work better for specific data patterns
   * Validate training data quality and representativeness
   * Check threshold selection (automatically set, but dataset-dependent)

For more examples, check the ``examples/`` directory in the `GitHub repository <https://github.com/spg-iitd/titli>`_.
