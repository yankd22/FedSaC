# FedSaC
Balancing Similarity and Complementarity for Unimodal and Multimodal Federated Learning

The implementation of FedSaC.

## Requirements
The needed libraries are in requirements.txt.

## Experiments
To run on CIFAR100, excute:

      python FedSaC.py --dataset CIFAR100 --partition 'noniid-skew' --n_parties 10

To run on CIFAR10, excute:

      python FedSaC.py --dataset CIFAR10 --partition 'noniid-skew' --n_parties 10

## Reference
The code structure is based on the code in [pFedGraph](https://github.com/MediaBrain-SJTU/pFedGraph).
