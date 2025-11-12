Feature Extractors (titli.fe)
==============================

The ``titli.fe`` module provides feature extraction capabilities for transforming raw network traffic into feature vectors suitable for machine learning models.

.. currentmodule:: titli.fe

Overview
--------

Feature extractors are responsible for converting network packet data (typically from PCAP files) into numerical feature vectors. These features capture various aspects of network traffic such as temporal patterns, statistical properties, and protocol-specific information.

Available Feature Extractors
-----------------------------

AfterImage
~~~~~~~~~~

.. autoclass:: AfterImage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

NetStat
~~~~~~~

.. autoclass:: NetStat
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Base Classes
------------

BaseTrafficFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: titli.fe.base_feature_extractor.BaseTrafficFeatureExtractor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

BaseFeatureExtractor
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: titli.fe.base_feature_extractor.BaseFeatureExtractor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Helper Functions
----------------

.. autofunction:: titli.fe.base_feature_extractor.load_dataset_info

Usage Examples
--------------

Basic Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

Extract features from a PCAP file and output to CSV for DataLoader usage:

.. code-block:: python

   from titli.fe import AfterImage
   
   fe = AfterImage(file_path="traffic.pcap")
   fe.extract_features(output_path="features.csv")

With Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from titli.fe import AfterImage
   
   fe = AfterImage(
       file_path="traffic.pcap",
       decay_factors=[5, 3, 1, 0.1, 0.01],
       max_pkt=100000,
       limit=10000
   )
   fe.extract_features(output_path="features.csv")

Integration with DataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use extracted features with DataLoader for model training:

.. code-block:: python

   from titli.fe import AfterImage
   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   
   # Extract features
   fe = AfterImage(file_path="traffic.pcap")
   fe.extract_features(output_path="features.csv")
   
   # Load with DataLoader
   dataset = StreamingCSVDataset(
       feature_csv_path="features.csv",
       label_csv_path="labels.csv",
       label_column=0
   )
   loader = DataLoader(dataset, batch_size=32)

.. note::

   Feature extractors output CSV files that should be consumed via ``StreamingCSVDataset`` 
   and ``DataLoader`` for model training. Direct use of extracted features is discouraged.
   State management is now handled internally by the feature extractors.
