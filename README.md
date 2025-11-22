# LSST-SL-Challenge

We are Team NAOC-AIchemists-A3 for [LSST-SL-Challenge](https://slchallenge.cbpf.br/).  
Our method is to train a ResNet on the simulation data with labels, and then adapt the network to blind test data without labels using Domain-Adversarial Neural Network (DANN).  
For more details on DANN, please refer to [Ganin2014](10.48550/arXiv.1409.7495) and [Ganin2015](10.48550/arXiv.1505.07818).  

## Scripts
`combined_classifier.py`: Base network built upon ResNet blocks.  
`dann_adaptation.py`: DANN implementation based on the fundamental network.  