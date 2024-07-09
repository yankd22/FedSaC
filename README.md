# FedSaC
Balancing Similarity and Complementarity for Federated Learning

The implementation of FedSaC.

## Requirements
The needed libraries are in requirements.txt.

## Experiments
To run on CIFAR100, excute:

      python FedSaC.py --dataset CIFAR100 --partition 'homo' --n_parties 10 --skew_class 20 --matrix_alpha 0.9 --matrix_beta 1.4

To run on CIFAR10, excute:

      python FedSaC.py --dataset CIFAR10 --partition 'homo' --n_parties 10  --skew_class 2 --matrix_alpha 0.9 --matrix_beta 1.4

## Reference
The code structure is based on the code in [pFedGraph](https://github.com/MediaBrain-SJTU/pFedGraph).
