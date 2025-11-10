Quick Start
===========

This guide will help you get started with Titli quickly.

Basic Example
-------------

Here's a simple example of training and testing a Kitsune IDS:

Training an IDS
~~~~~~~~~~~~~~~

.. code-block:: python

   from titli.fe import AfterImage
   from titli.ids import KitsuneIDS
   from titli.utils import StreamingCSVDataset
   
   # Initialize feature extractor
   feature_extractor = AfterImage(
       file_path="path/to/training.pcap",
       dataset_name="my_dataset"
   )
   
   # Extract features from training data
   features = feature_extractor.extract_features()
   
   # Initialize and train the IDS
   ids = KitsuneIDS(
       dataset_name="my_dataset",
       input_size=features.shape[1],
       max_autoencoder_size=10,
       FM_grace_period=10000,
       AD_grace_period=50000
   )
   
   # Train the model
   ids.train_model(train_loader)

Testing an IDS
~~~~~~~~~~~~~~

.. code-block:: python

   from titli.fe import AfterImage
   from titli.ids import KitsuneIDS
   
   # Load the trained model
   ids = KitsuneIDS.load_model("path/to/saved_model.pkl")
   
   # Initialize feature extractor for test data
   feature_extractor = AfterImage(
       file_path="path/to/test.pcap",
       dataset_name="my_dataset"
   )
   
   # Extract features from test data
   test_features = feature_extractor.extract_features()
   
   # Perform inference
   anomaly_scores = ids.infer(test_features)
   
   # Evaluate results
   predictions = (anomaly_scores > threshold).astype(int)

Understanding the Pipeline
--------------------------

Titli follows a standard pipeline for IDS:

1. **Feature Extraction**: Convert raw network traffic into feature vectors
2. **Model Training**: Train an anomaly detection model on normal traffic
3. **Inference**: Detect anomalies in new traffic
4. **Evaluation**: Measure model performance with various metrics

Feature Extractors
~~~~~~~~~~~~~~~~~~

Titli provides several feature extractors:

* **AfterImage**: Packet-based feature extractor used in Kitsune
* **NetStat**: Network statistics extractor

IDS Models
~~~~~~~~~~

Titli includes various IDS models:

* **KitNET**: Ensemble of autoencoders for anomaly detection
* **LOF**: Local Outlier Factor
* **OCSVM**: One-Class Support Vector Machine
* **Autoencoder**: Deep learning-based autoencoder
* **VAE**: Variational Autoencoder
* **ICL**: Incremental Correlation Learning

Working with Different Data Sources
------------------------------------

PCAP Files
~~~~~~~~~~

.. code-block:: python

   from titli.fe import AfterImage
   
   # Extract features from PCAP file
   fe = AfterImage(file_path="traffic.pcap", dataset_name="test")
   features = fe.extract_features()

CSV Files
~~~~~~~~~

.. code-block:: python

   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   
   # Load features from CSV
   dataset = StreamingCSVDataset("features.csv")
   loader = DataLoader(dataset, batch_size=32, shuffle=False)

Next Steps
----------

* Read the :doc:`usage` guide for more detailed examples
* Explore the :doc:`api/fe` documentation for feature extractors
* Check the :doc:`api/ids` documentation for available IDS models
* View the example scripts in the ``examples/`` directory of the repository
