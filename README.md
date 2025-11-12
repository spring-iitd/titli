<img src="assets/images/ titli-logo.png" alt="Titli Logo" width="150" />


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/titli)
![PyPI - Version](https://img.shields.io/pypi/v/titli)
![GitHub License](https://img.shields.io/github/license/spg-iitd/titli)
[![Documentation Status](https://readthedocs.org/projects/titli/badge/?version=latest)](https://titli.readthedocs.io/en/latest/?badge=latest)

A toolkit for hosting feature extraction, model training, model inference, and model evaluation of AI-based Intrusion Detection Systems
<p align="center">
	<img src="assets/images/pipeline-overview.jpg" alt="Pipeline Overview" width="800" />
</p>

## Documentation

📚 **[Read the full documentation](docs/)** to get started with Titli.

The documentation includes:
- Installation guide
- Quick start tutorial
- Detailed usage examples
- Complete API reference
- And more!

To build the documentation locally:
```bash
cd docs
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
make html
```

Then open `docs/build/html/index.html` in your web browser.

### Installation
```
pip install titli
```

### Usage
- Step 1: Copy the ```test_titli.ipynb``` file and run it locally.
- Step 2: Run both the files to train and test IDSs.

<!-- ### Todo (Developer Tasks)
- [ ] Check if RMSE is used for loss or just the difference.
- [ ] Put Kitsune code into the base IDS format.
- [ ] Write code to evaluate the model and calculate all the metrics.  

### TODO New:
- [ ] Resolve BaseSKLearn class infer function's TODO
- [ ] Make fit and predict function as private by putting "_" in the starting
- [ ] Similar to PyTorch, define __call__ function for SkLearn models too
- [ ] Investigate and modify the use of get_model in the Pytorch class.
- [ ] Make internal methods private
- [ ] Add docstrings in the class and functions

- Write where the model is saved in the print statement! -->