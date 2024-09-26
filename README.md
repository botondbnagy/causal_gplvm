# Bivariate Causal Discovery: Bayesian Model Selection and Cross-validation

This project was carried out during my summer internship and is built on code from the authors of [this paper](https://arxiv.org/abs/2306.02931).

### Installation

To install dependencies, run `pip install -r requirements.txt`

### File structure:

- `experiments/` contains scripts to run the main experiments (cross validation and infinite data) with real and synthetic data

- `data/` contains the data files and additional modules

- `utils.py` contains utility functions

- `models/` contains the GPLVM models

	- BayesGPLVM: Unsupervised closed form GPLVM
	
	- PartObsBayesianGPLVM: Conditional closed form GPLVM
	
	- GeneralisedUnsupGPLVM: Unsupervised stochastic GPLVM (large data)
	
	- GeneralisedGPLVM: Conditional stochastic GPLVM (large data)

- `methods/` contains the scripts required for fitting the GPLVM and outputting marginal likelihoods

- `results/` contains all the results of the experiments (not public)

- `figs/` contains figures illustrating the main findings of the project

- `hpc_runs/` contains GPU job submission scripts for the High Performance Cluster


### Cross-validation:

Cross-validation was compared with the Bayesian model selection method. The difference in marginal likelihoods between causal directions was found to be significant enough to warrant the use of cross-validation in additive noise models:

![Cross-validation](figs/AN_causal_stem.jpg)

### Infinite Data:

The infinite data experiment was carried out to test the performance of the causal discovery methods in the limit of infinite data. The quality of the GPLVM fit was found to increase with more data, as expected, however this is an ongoing experiment with more complex datasets to be tested for a comprehensive conclusion.

#### Latent Gaussian Process Models:

Performance was found to increase with better inference, better initialisation of hyperparameters, as well as inclusion of more kernels. As this is an active area of research, we expect implementation of any improvements in these to help.

#### Random Restarts:

The overarching principle behind the method is to choose the model with highest marginal likelihood (or lower bound to it). As latent GP models are susceptible to local optima, random restarts with different initialisation is recommended.
