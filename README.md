# Titli
A toolkit for hosting feature extraction, model training, model inference, and model evaluation of AI-based Intrusion Detection Systems
<p align="center">
	<img src="assets/images/pipeline-overview.jpg" alt="Pipeline Overview" width="800" />
</p>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/titli)
![PyPI - Version](https://img.shields.io/pypi/v/titli)
![GitHub License](https://img.shields.io/github/license/spg-iitd/titli)

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
- Step 1: Copy the ```examples/train_ids.py``` and ```examples/test_ids.py``` file from the repo to your local machine.
- Step 2: Run both the files to train and test the Kitsune IDS respectively.

### Todo (Developer Tasks)
- [ ] Check if RMSE is used for loss or just the difference.
- [ ] Put Kitsune code into the base IDS format.
- [ ] Write code to evaluate the model and calculate all the metrics.  

### TODO New:
- [ ] Resolve BaseSKLearn class infer function's TODO
- [ ] Make fit and predict function as private by putting "_" in the starting
- [ ] Similar to PyTorch, define __call__ function for SkLearn models too

- Write where the model is saved in the print statement!