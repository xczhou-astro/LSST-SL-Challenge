# LSST-SL-Challenge

We are Team NAOC-AIchemists-A3 for [LSST-SL-Challenge](https://slchallenge.cbpf.br/).  
Our method is to train a ResNet on the simulation data with labels, and then adapt the network to blind test data without labels using Domain-Adversarial Neural Network (DANN).  
For more details on DANN, please refer to [Ganin & Lempitsky 2014](https://arxiv.org/abs/1409.7495) and [Ganin et al. 2015](https://arxiv.org/abs/1505.07818).  

## Scripts
`combined_classifier.py`: Fundamental network built upon ResNet blocks.  
`dann_adaptation.py`: DANN implementation based on the fundamental network.  
`get_dann_embs.py`: Get DANN embeddings.  

## Data
DANN embeddings and 2d UMAP embeddings can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1pRNXeJ0HOlsAUx2DNQB7wQas4zBpJ26f?usp=drive_link)

The processed numpy files used for training and the processing code can be requested by email to [Xingchen Zhou](mailto:xczhou95@gmail.com)