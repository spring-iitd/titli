# Titli
Artificial Intelligence based Intrusion Detection Systems

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/titli)
![PyPI - Version](https://img.shields.io/pypi/v/titli)
![GitHub License](https://img.shields.io/github/license/spg-iitd/titli)

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