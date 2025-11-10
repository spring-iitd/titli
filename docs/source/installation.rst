Installation
============

Requirements
------------

Titli requires Python 3.10 or higher.

Dependencies
~~~~~~~~~~~~

* scapy
* numpy
* pandas
* matplotlib
* scikit-learn
* scipy
* tqdm
* torch
* torchvision

Installing from PyPI
--------------------

The easiest way to install Titli is using pip:

.. code-block:: bash

   pip install titli

This will install Titli and all required dependencies except PyTorch. 

Installing PyTorch
~~~~~~~~~~~~~~~~~~

For optimal performance, especially if you have GPU support, it's recommended to install PyTorch separately 
according to your system configuration. Visit the `PyTorch website <https://pytorch.org/get-started/locally/>`_ 
for installation instructions specific to your platform.

For example, to install PyTorch with CUDA support:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CPU only
   pip install torch torchvision

Installing from Source
----------------------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/spg-iitd/titli.git
   cd titli
   pip install -e .

This installs Titli in editable mode, allowing you to modify the source code and see changes immediately.

Verifying Installation
----------------------

To verify that Titli is installed correctly, you can run:

.. code-block:: python

   import titli
   from titli.fe import AfterImage
   from titli.ids import KitNET
   
   print("Titli installed successfully!")

If no errors occur, the installation was successful.

Development Installation
------------------------

If you plan to contribute to Titli, install the development dependencies:

.. code-block:: bash

   git clone https://github.com/spg-iitd/titli.git
   cd titli
   pip install -e ".[dev]"

This includes additional tools for testing, linting, and building documentation.
