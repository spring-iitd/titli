Utilities (titli.utils)
=======================

The ``titli.utils`` module provides utility functions and classes for data handling, loss computation, and dataset management.

.. currentmodule:: titli.utils

Overview
--------

This module contains helper utilities that support the main feature extraction and IDS functionality.

Datasets
--------

StreamingCSVDataset
~~~~~~~~~~~~~~~~~~~

.. autoclass:: StreamingCSVDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Loss Functions
--------------

RMSELoss
~~~~~~~~

.. autoclass:: RMSELoss
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Data Utilities
--------------

Directory Management
~~~~~~~~~~~~~~~~~~~~

.. automodule:: titli.utils.data
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Using StreamingCSVDataset
~~~~~~~~~~~~~~~~~~~~~~~~~

Load and iterate over CSV data:

.. code-block:: python

   from titli.utils import StreamingCSVDataset
   from torch.utils.data import DataLoader
   
   # Create dataset
   dataset = StreamingCSVDataset(
       csv_path="features.csv",
       label_column="label"
   )
   
   # Create data loader
   loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=False
   )
   
   # Iterate over batches
   for features, labels in loader:
       # Process batch
       pass

Using RMSELoss
~~~~~~~~~~~~~~

Use RMSE as a loss function in PyTorch:

.. code-block:: python

   import torch
   from titli.utils import RMSELoss
   
   # Initialize loss function
   criterion = RMSELoss()
   
   # Compute loss
   predictions = model(inputs)
   loss = criterion(predictions, targets)
   
   # Backward pass
   loss.backward()

Directory Management
~~~~~~~~~~~~~~~~~~~~

Create organized directory structures for experiments:

.. code-block:: python

   from titli.utils.data import create_directories
   
   # Create standard directory structure
   create_directories("my_experiment")
   
   # This creates:
   # - my_experiment/models/
   # - my_experiment/results/
   # - my_experiment/logs/

Advanced Usage
--------------

Custom Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Process large CSV files in chunks:

.. code-block:: python

   from titli.utils import StreamingCSVDataset
   import pandas as pd
   
   # Create streaming dataset
   dataset = StreamingCSVDataset(
       csv_path="large_file.csv",
       chunk_size=10000
   )
   
   # Process in chunks to avoid memory issues
   for i, (features, labels) in enumerate(dataset):
       print(f"Processing chunk {i}")
       # Your processing logic here

Custom Loss Functions
~~~~~~~~~~~~~~~~~~~~~

Combine with other PyTorch loss functions:

.. code-block:: python

   import torch
   import torch.nn as nn
   from titli.utils import RMSELoss
   
   class CombinedLoss(nn.Module):
       def __init__(self, alpha=0.5):
           super().__init__()
           self.rmse = RMSELoss()
           self.mse = nn.MSELoss()
           self.alpha = alpha
       
       def forward(self, pred, target):
           return self.alpha * self.rmse(pred, target) + \
                  (1 - self.alpha) * self.mse(pred, target)
   
   criterion = CombinedLoss(alpha=0.7)
