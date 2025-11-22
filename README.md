# LSST-SL-Challenge

We are Team NAOC-AIchemists-A3 for [LSST-SL-Challenge](https://slchallenge.cbpf.br/).  
Our method is to train a ResNet on the simulation data with labels, and then adapt the network to blind test data without labels using Domain-Adversarial Neural Network (DANN).  
For more details on DANN, please refer to [Ganin2014](https://arxiv.org/abs/1409.7495) and [Ganin2015](https://arxiv.org/abs/1505.07818).  

## Scripts
`combined_classifier.py`: Fundamental network built upon ResNet blocks.  
`dann_adaptation.py`: DANN implementation based on the fundamental network.  