# About
This repository contains codes to reproduce experimental results of the paper:

E.Shimizu, K.Fukumizu, D.Sejdinovic, **Neural-Kernel Conditioal Mean Embeddings**. In *Proceedings of International Conference on Machine Learning* (ICML), 2024.

Install the latest versions of:
mlflow, hydra, tqdm, jax, optax, equinox, gymnax, normflows, lightning-uq-box

**NN-CME_example.ipynb** contains a short demonstration of our proposal on a toy data setting.

**CDE** file contains codes for the density estimation experiments. For example, run codes like:
```
python toy_Proposal_Joint.py data.name="bimodal" mlflow.runname="bimodal" 
```
```
python uci_Proposal_Joint.py data.data_path="bostonHousing/data" mlflow.runname="boston"
```

UCI datasets can be downloaded from the Github repository of the [CARD (Han et al., 2022)](https://github.com/XzwHan/CARD/tree/main):
Place each dataset inside CDE/dataset


**RL** file contains codes for the RL experiments. For example, run codes like:
```
python DQN.py env_name="CartPole-v1" optimizer.lr=1e-4 optimizer.eps=1e-5 mlflow.runname="CartPole"
```
