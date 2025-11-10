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
   
   # Initialize the feature extractor
   fe = AfterImage(
       file_path="path/to/traffic.pcap",
       dataset_name="my_dataset",
       limit=float("inf"),  # Maximum number of records
       decay_factors=[5, 3, 1, 0.1, 0.01],  # Time windows
       max_pkt=float("inf")  # Maximum packets to process
   )
   
   # Extract features
   features = fe.extract_features()
   
   # Save state for later use
   fe.save_state("feature_extractor_state.pkl")

Advanced Usage: Stateful Feature Extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can save the state of the feature extractor after processing training data and reuse it for test data:

.. code-block:: python

   from titli.fe import AfterImage, NetStat
   import pickle
   
   # Train phase: Extract features and save state
   train_fe = AfterImage(
       file_path="training.pcap",
       dataset_name="train"
   )
   train_features = train_fe.extract_features()
   
   # Save the network state
   with open("netstat_state.pkl", "wb") as f:
       pickle.dump(train_fe.state, f)
   
   # Test phase: Load state and extract features
   with open("netstat_state.pkl", "rb") as f:
       saved_state = pickle.load(f)
   
   test_fe = AfterImage(
       file_path="test.pcap",
       dataset_name="test",
       state=saved_state  # Reuse the saved state
   )
   test_features = test_fe.extract_features()

Training IDS Models
-------------------

KitNET (Kitsune)
~~~~~~~~~~~~~~~~

KitNET is an ensemble of autoencoders designed for online anomaly detection.

.. code-block:: python

   from titli.ids import KitsuneIDS
   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   
   # Prepare data loader
   train_dataset = StreamingCSVDataset("train_features.csv")
   train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
   
   # Initialize KitNET
   ids = KitsuneIDS(
       dataset_name="my_dataset",
       input_size=100,  # Number of features
       max_autoencoder_size=10,
       FM_grace_period=10000,  # Feature mapping period
       AD_grace_period=50000,  # Anomaly detection training period
       learning_rate=0.1,
       hidden_ratio=0.75
   )
   
   # Train the model
   ids.train_model(train_loader)
   
   # Save the trained model
   ids.save_model("kitsune_model.pkl")

PyTorch-Based Kitsune
~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration, use the PyTorch implementation:

.. code-block:: python

   from titli.ids import PyTorchKitsune
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   ids = PyTorchKitsune(
       dataset_name="my_dataset",
       input_size=100,
       device=device
   )
   
   ids.train_model(train_loader)

Scikit-learn Based Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Local Outlier Factor (LOF)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from titli.ids import LOF
   from sklearn.preprocessing import StandardScaler
   
   # Initialize LOF
   lof = LOF(
       dataset_name="my_dataset",
       input_size=100,
       device="cpu",
       n_neighbors=20,
       contamination=0.1
   )
   
   # Train the model
   lof.train_model(train_loader)

One-Class SVM
^^^^^^^^^^^^^

.. code-block:: python

   from titli.ids import OCSVM
   
   # Initialize OCSVM
   ocsvm = OCSVM(
       dataset_name="my_dataset",
       input_size=100,
       device="cpu",
       kernel='rbf',
       gamma='auto',
       nu=0.1
   )
   
   # Train the model
   ocsvm.train_model(train_loader)

Deep Learning Models
~~~~~~~~~~~~~~~~~~~~

Autoencoder
^^^^^^^^^^^

.. code-block:: python

   from titli.ids import Autoencoder
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Initialize Autoencoder
   ae = Autoencoder(
       dataset_name="my_dataset",
       input_size=100,
       hidden_size=50,
       device=device,
       learning_rate=0.001,
       num_epochs=50
   )
   
   # Train the model
   ae.train_model(train_loader)

Variational Autoencoder (VAE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from titli.ids import VAE
   
   # Initialize VAE
   vae = VAE(
       dataset_name="my_dataset",
       input_size=100,
       hidden_size=50,
       latent_size=20,
       device=device
   )
   
   # Train the model
   vae.train_model(train_loader)

Inference and Evaluation
-------------------------

Making Predictions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load test data
   test_dataset = StreamingCSVDataset("test_features.csv")
   test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
   
   # Load trained model
   ids = KitsuneIDS.load_model("kitsune_model.pkl")
   
   # Perform inference
   anomaly_scores, predictions = ids.infer(test_loader)
   
   # anomaly_scores: continuous anomaly scores
   # predictions: binary predictions (0=normal, 1=anomaly)

Evaluating Models
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
   
   # Assuming you have true labels
   true_labels = test_dataset.labels
   
   # Calculate metrics
   accuracy = accuracy_score(true_labels, predictions)
   precision = precision_score(true_labels, predictions)
   recall = recall_score(true_labels, predictions)
   f1 = f1_score(true_labels, predictions)
   
   print(f"Accuracy: {accuracy:.4f}")
   print(f"Precision: {precision:.4f}")
   print(f"Recall: {recall:.4f}")
   print(f"F1-Score: {f1:.4f}")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Plot anomaly scores
   plt.figure(figsize=(12, 4))
   plt.plot(anomaly_scores)
   plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
   plt.xlabel('Sample')
   plt.ylabel('Anomaly Score')
   plt.title('Anomaly Detection Results')
   plt.legend()
   plt.show()

Advanced Topics
---------------

Custom Feature Extractors
~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom feature extractor, inherit from ``BaseTrafficFeatureExtractor``:

.. code-block:: python

   from titli.fe.base_feature_extractor import BaseTrafficFeatureExtractor
   
   class MyFeatureExtractor(BaseTrafficFeatureExtractor):
       def __init__(self, file_path, dataset_name=None, **kwargs):
           super().__init__(file_path=file_path, dataset_name=dataset_name, **kwargs)
           # Initialize your custom parameters
       
       def extract_features(self):
           # Implement your feature extraction logic
           pass

Working with Streaming Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For real-time or streaming scenarios:

.. code-block:: python

   from titli.utils import StreamingCSVDataset
   
   # Create a streaming dataset
   dataset = StreamingCSVDataset(
       csv_path="features.csv",
       chunk_size=1000  # Process in chunks
   )
   
   # Process in batches
   for batch in dataset:
       features, labels = batch
       # Process the batch

Data Preprocessing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   import numpy as np
   
   # Normalize features
   scaler = StandardScaler()
   normalized_features = scaler.fit_transform(features)
   
   # Save scaler for later use
   import pickle
   with open("scaler.pkl", "wb") as f:
       pickle.dump(scaler, f)

Tips and Best Practices
------------------------

1. **Grace Periods**: Set appropriate grace periods for KitNET based on your dataset size
2. **Feature Scaling**: Always normalize/standardize features for better model performance
3. **Threshold Selection**: Use validation data to select optimal anomaly thresholds
4. **State Management**: Save and reuse feature extractor states for consistent testing
5. **GPU Acceleration**: Use PyTorch-based models when you have GPU access
6. **Batch Processing**: Process large datasets in batches to avoid memory issues

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**PCAP File Not Found**
   Ensure the path to your PCAP file is correct and the file exists.

**Out of Memory**
   Reduce batch size or process data in chunks using streaming datasets.

**Model Not Converging**
   Adjust learning rate, increase training epochs, or check data quality.

**Poor Detection Performance**
   * Ensure proper feature normalization
   * Verify grace periods are sufficient
   * Check threshold selection
   * Validate training data quality

For more examples, check the ``examples/`` directory in the `GitHub repository <https://github.com/spg-iitd/titli>`_.
