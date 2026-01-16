import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import subprocess
import math
import cuml
from tqdm import tqdm

from combined_classifier import DeterministicResNetClassifier, BasicBlock
from dann_adaptation import BlindTestDataset, DANNModelTransfer

def get_features(feature_extractor, data_loader, class_classifier=None, device='cuda'):
    
    feature_extractor.to(device)
    feature_extractor.eval()
    
    if class_classifier is not None:
        class_classifier.to(device)
        class_classifier.eval()
    
    
    features = []
    filenames = []
    scores = []
    
    with torch.no_grad():
        for data, names in tqdm(data_loader, desc='Getting features', total=len(data_loader)):
            data = data.to(device)
            feat = feature_extractor(data)
            
            if class_classifier is not None:
                score = class_classifier(feat)
            
            features.append(feat.cpu().numpy())
            scores.append(score.cpu().numpy())
            filenames.append(names)
            
    features = np.concatenate(features, axis=0)
    filenames = np.concatenate(filenames, axis=0)
    scores = np.concatenate(scores, axis=0)
    return features, filenames, scores

if __name__ == '__main__':
    
    with open('results_dann_adaptation_to_blind_test_aug/data_splits_seed_42.json', 'r') as f:
        splits = json.load(f)
            
    source_train_data_ls = splits['train_source']
    source_val_data_ls = splits['val_source']
    blind_test_data_ls = splits['blind_test']
    
    source_train_dataset = BlindTestDataset(source_train_data_ls)
    source_val_dataset = BlindTestDataset(source_val_data_ls)
    blind_test_dataset = BlindTestDataset(blind_test_data_ls)
    
    source_train_loader = DataLoader(source_train_dataset, batch_size=512,
                                    shuffle=False, num_workers=4)
    source_val_loader = DataLoader(source_val_dataset, batch_size=512,
                                shuffle=False, num_workers=4)
    blind_test_loader = DataLoader(blind_test_dataset, batch_size=512,
                                shuffle=False, num_workers=4)
    
    base_model = DeterministicResNetClassifier(BasicBlock, [2, 2, 2, 2])
    
    model = DANNModelTransfer(base_model)
    
    weights = torch.load('results_dann_adaptation_to_blind_test_aug/dann_model_best.pth', 
                    map_location='cuda')
    
    model.load_state_dict(weights['model_state_dict'])
    
    os.makedirs('dann_embeddings', exist_ok=True)
    
    source_train_embs, source_train_filenames, source_train_scores = get_features(
        model.feature_extractor, source_train_loader, model.class_classifier, device='cuda')
    
    source_val_embs, source_val_filenames, source_val_scores = get_features(
        model.feature_extractor, source_val_loader, model.class_classifier, device='cuda')
    
    blind_test_embs, blind_test_filenames, blind_test_scores = get_features(
        model.feature_extractor, blind_test_loader, model.class_classifier, device='cuda')
    
    np.savez('dann_embeddings/source_train_embs.npz',
            embeddings=source_train_embs, filenames=source_train_filenames, 
            scores=source_train_scores)
    np.savez('dann_embeddings/source_val_embs.npz',
            embeddings=source_val_embs, filenames=source_val_filenames,
            scores=source_val_scores)
    np.savez('dann_embeddings/blind_test_embs.npz',
            embeddings=blind_test_embs, filenames=blind_test_filenames, 
            scores=blind_test_scores)