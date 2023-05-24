# Disentangling Causal Mechanisms By Obstructing Classifiers


## 1. Environment Setup

Tested with python 3.9.12 

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt 
```


## 2. Dataset Setup 

We used colored MNIST with each number randomly chosed to be red, green, or blue. Further dataset development may be conducted in the dataset module. We use the proportions of [0.8, 0.1, 0.1] for train, validation, test split. 

```
python dataset/dataset_utils.py
```


## 3. Module Summary

*Dataset*:
`dataset/dataset_utils.py` provides useful utilities to (1) create our dataset, (2) load our dataset, and (3) run the train/validation/test split.

*Models*:
`models/` stores our models. Within this folder, we have a convolutional classifier, a logistic classifier (i.e. one-layered sigmoid head), a convolutional autoencoder, and a convolutional autoencoder with projection built in.

*Models/Interventions*:
`models/interventions/` holds our interventions on models. Specifically, we have `rlace.py` which linearly projects a rank-k dimension subspace and we have `counterfactual.py` which uses gradient descent on the activations to produce a counterfactual (similar to adversarial techniques, but works in the latent space).
Though the file structures are a little bit different, `run_inlp.py` and `run_wtmt.py` run the given interventions (over each concept) and test the subsequent models (over each concept).


## 4. Run code 

Runs the baseline and disentangled experiments. 
```bash
python baseline_experiments.py --baseline-classification --baseline-autoencoder
python disentangled_learning_experiments.py
```

Note: Most of the figures in the paper were generated from the `colored-mnist-experiments.ipynb` notebook

For more, information, please read our [paper](https://github.com/surajK610/disentangled-learning-by-projection/blob/main/Disentangling_Causal_Mechanisms_By_Obstructing_Classifiers.pdf)