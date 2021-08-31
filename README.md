# MLDS2: Reproducibility of AFN

This is the code repository for the reproducibility project. The topic of the reproduction is the article [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://arxiv.org/pdf/1909.03276.pdf)

The repository is organized as follows:
* Directory `models`, which contains:
    * `afn.py` - proposed model implementation,
    * `lr.py` - logistic regression model implementation,
    * `fm.py` - factorization machine model implementation,
    * `deepfm.py` - DeepFM model implementation.
* Jupyter notebook `Evaluate_Models.ipynb` used for running the training and evaluation procedures,
* Directory `transform`, which contains scripts to transform the LIBSVM datasets from the original repository into the appropriate format for the above models,
    * These scripts need to be ran before any model training!
* `requirements.txt` used to initialize the conda environment.

The environment is initialized and activated as follows:
```
conda create --name repr --file requirements.txt
conda activate repr
```
