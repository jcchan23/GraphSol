# GraphSol
A Protein Solubility Predictor developed by Graph Convolutional Network and Predicted Contact Map

The source code for our paper [Structure-aware protein solubility prediction from sequence through graph convolutional network and predicted contact map](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00488-1)

Please use [SPOT-Contact](https://sparks-lab.org/downloads/) to generate all the features. (This step will take most of the time.)

1D-features: BLOSUM, PSSM, HMM, SPIDER3, AAPHY7

2D-features: SPOTCON

These six files could be generate two matrices files by using get1D_features.py and get2D_features.py, which will be use as the train.py

We will complete all the details later.

(Under developed...)
