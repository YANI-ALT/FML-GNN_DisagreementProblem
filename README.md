# FML-GNN_DisagreementProblem

This repository contains the code for studying the disagreement problem for Graph Neural Networks.

### Installation
After cloning the repo, the libraries required to run the code are listed in the environment.yml file.


### Running the code
The `train_models.py` file can be used to train GNN models from the command line.

    python train_models.py --model GCN --dataset Cora

The `generated_explanations.py` python script will generate and save explanations.

    python generated_explanations.py

The `generate_disagreement.py` python script will generate all the disagreement metrics and plots for the explanations saved.

    python generate_disagreement.py


The `Tutorial1.ipynb` file contains the example code for training the GNN models.

The `Tutorial2.ipynb` file contains the example code for loading a pretrained GNN model using checkpoints.

The `Models_Summary.ipynb` file contains information on the models (test/train accuracy) and dataset


### Reference Links :
1. [DIG Library](https://github.com/divelab/DIG)
2. [Explainability in Graph Neural Networks: A Taxonomic Survey](https://arxiv.org/pdf/2012.15445.pdf)
3. [The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective](https://arxiv.org/abs/2202.01602)
4. [Pytorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html)
5. Graph XAI (https://github.com/mims-harvard/GraphXAI)



----

This project was done by [Anubhav Goel](https://github.com/anubhavgoel26) , [Devyani Maladkar](https://github.com/YANI-ALT), [Shray Mathur](https://github.com/Shray64) and [Shagun Gupta](https://github.com/Shagun-G), as part of the Fair and Transparent Machine Learning Course at UT Austin. 
