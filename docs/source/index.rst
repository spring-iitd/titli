Titli Documentation
===================

**Titli** is a comprehensive toolkit for hosting feature extraction, model training, model inference, 
and model evaluation of AI-based Intrusion Detection Systems (IDS).

.. image:: ../../assets/images/pipeline-overview.jpg
   :alt: Pipeline Overview
   :width: 800px
   :align: center

|

.. image:: https://img.shields.io/pypi/pyversions/titli
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/v/titli
   :alt: PyPI - Version

.. image:: https://img.shields.io/github/license/spg-iitd/titli
   :alt: GitHub License

Overview
--------

Titli provides a modular framework for building and evaluating intrusion detection systems with a unified API. It includes:

* **6 IDS Models**: LOF, OCSVM, VAE, Autoencoder, ICL, KitNET
* **Unified API**: All models expose 5 consistent methods (train_model, save, load, infer, evaluate)
* **Efficient DataLoaders**: StreamingCSVDataset for large-scale data processing
* **Comprehensive Evaluation**: Automatic metrics computation and visualization
* **Easy Persistence**: Simple save/load model management with default paths
* **Feature Extractors**: Tools for extracting features from network traffic (AfterImage, NetStat)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference
   api/fe
   api/ids
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
