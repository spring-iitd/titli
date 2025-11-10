Intrusion Detection Systems (titli.ids)
=======================================

The ``titli.ids`` module provides various intrusion detection system (IDS) models for anomaly detection in network traffic.

.. currentmodule:: titli.ids

Overview
--------

This module includes both traditional machine learning models (LOF, OCSVM) and deep learning models (Autoencoders, VAE) for anomaly detection. Each model follows a consistent interface for training, inference, and evaluation.

Ensemble Models
---------------

KitNET
~~~~~~

.. autoclass:: KitNET
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

KitsuneIDS
~~~~~~~~~~

.. autoclass:: KitsuneIDS
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

TorchKitNET
~~~~~~~~~~~

.. autoclass:: TorchKitNET
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PyTorchKitsune
~~~~~~~~~~~~~~

.. autoclass:: PyTorchKitsune
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Traditional ML Models
---------------------

LOF (Local Outlier Factor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LOF
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

OCSVM (One-Class SVM)
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OCSVM
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Deep Learning Models
--------------------

Autoencoder
~~~~~~~~~~~

.. autoclass:: Autoencoder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

VAE (Variational Autoencoder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VAE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Other Models
------------

ICL (Incremental Correlation Learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ICL
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Base Classes
------------

BasePyTorchModel
~~~~~~~~~~~~~~~~

.. autoclass:: titli.ids.base_ids.BasePyTorchModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

BaseSKLearnModel
~~~~~~~~~~~~~~~~

.. autoclass:: titli.ids.base_ids.BaseSKLearnModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

KitNET Example
~~~~~~~~~~~~~~

.. code-block:: python

   from titli.ids import KitsuneIDS
   from torch.utils.data import DataLoader
   
   # Initialize model
   ids = KitsuneIDS(
       dataset_name="my_dataset",
       input_size=100,
       max_autoencoder_size=10,
       FM_grace_period=10000,
       AD_grace_period=50000
   )
   
   # Train
   ids.train_model(train_loader)
   
   # Save
   ids.save_model("model.pkl")
   
   # Inference
   scores, predictions = ids.infer(test_loader)

LOF Example
~~~~~~~~~~~

.. code-block:: python

   from titli.ids import LOF
   
   lof = LOF(
       dataset_name="my_dataset",
       input_size=100,
       device="cpu",
       n_neighbors=20
   )
   
   lof.train_model(train_loader)
   scores, predictions = lof.infer(test_loader)

Autoencoder Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from titli.ids import Autoencoder
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   ae = Autoencoder(
       dataset_name="my_dataset",
       input_size=100,
       hidden_size=50,
       device=device,
       learning_rate=0.001,
       num_epochs=50
   )
   
   ae.train_model(train_loader)
   scores, predictions = ae.infer(test_loader)

Model Comparison
----------------

+------------------+------------------+------------------+------------------+
| Model            | Type             | GPU Support      | Online Learning  |
+==================+==================+==================+==================+
| KitNET           | Ensemble         | No               | Yes              |
+------------------+------------------+------------------+------------------+
| PyTorchKitsune   | Ensemble         | Yes              | Yes              |
+------------------+------------------+------------------+------------------+
| LOF              | Traditional ML   | No               | No               |
+------------------+------------------+------------------+------------------+
| OCSVM            | Traditional ML   | No               | No               |
+------------------+------------------+------------------+------------------+
| Autoencoder      | Deep Learning    | Yes              | No               |
+------------------+------------------+------------------+------------------+
| VAE              | Deep Learning    | Yes              | No               |
+------------------+------------------+------------------+------------------+
| ICL              | Correlation      | No               | Yes              |
+------------------+------------------+------------------+------------------+

Model Selection Guide
---------------------

* **KitNET/PyTorchKitsune**: Best for online anomaly detection with high-dimensional data
* **LOF**: Good for small to medium datasets with clear outliers
* **OCSVM**: Suitable for datasets with clear decision boundaries
* **Autoencoder/VAE**: Best for complex patterns and when GPU is available
* **ICL**: Useful for incremental learning scenarios
