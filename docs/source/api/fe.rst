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

Extract features from a PCAP file:

.. code-block:: python

   from titli.fe import AfterImage
   
   fe = AfterImage(
       file_path="traffic.pcap",
       dataset_name="my_dataset"
   )
   features = fe.extract_features()

With Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from titli.fe import AfterImage
   
   fe = AfterImage(
       file_path="traffic.pcap",
       dataset_name="my_dataset",
       decay_factors=[5, 3, 1, 0.1, 0.01],
       max_pkt=100000,
       limit=10000
   )
   features = fe.extract_features()

Stateful Extraction
~~~~~~~~~~~~~~~~~~~

Save and reuse the feature extractor state:

.. code-block:: python

   import pickle
   from titli.fe import AfterImage
   
   # Training phase
   train_fe = AfterImage(file_path="train.pcap", dataset_name="train")
   train_features = train_fe.extract_features()
   
   with open("state.pkl", "wb") as f:
       pickle.dump(train_fe.state, f)
   
   # Test phase
   with open("state.pkl", "rb") as f:
       state = pickle.load(f)
   
   test_fe = AfterImage(
       file_path="test.pcap",
       dataset_name="test",
       state=state
   )
   test_features = test_fe.extract_features()
