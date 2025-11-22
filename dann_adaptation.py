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
import cuml
from combined_classifier import BasicBlock, DeterministicResNetClassifier


class Tee:
    """Class to duplicate output to both stdout and a file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Force write to file immediately
    
    def flush(self):
        for f in self.files:
            f.flush()

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class BlindTestDataset(Dataset):
    """Dataset class for blind test data without labels"""
    
    def __init__(self, data_ls, augmentation_types=None):
        self.data_ls = data_ls
        self.augmentation_types = augmentation_types
        
    def __len__(self):
        return len(self.data_ls)
    
    def _apply_augmentation(self, image, aug_type):
        # image is in (5, 41, 41)
        if aug_type.startswith('rotate_'):
            angle = int(aug_type.split('_')[1])
            k = int(angle / 90)
            image = np.rot90(image, k=k, axes=(1, 2))
            
        elif aug_type == 'flip_horizontal':
            image = np.flip(image, axis=2)
            
        elif aug_type == 'flip_vertical':
            image = np.flip(image, axis=1)
            
        elif aug_type == 'flip_diagonal':
            image = np.transpose(image, (0, 2, 1))
            
        elif aug_type == 'flip_anti-diagonal':
            image = np.flip(np.transpose(image, (0, 2, 1)), axis=2)
        
        return image
    
    def __getitem__(self, idx):
        filename = self.data_ls[idx]
        basename = os.path.basename(filename)
        data = np.load(filename)
        
        if self.augmentation_types is not None:
            aug_type = self.augmentation_types[idx]
            if aug_type != 'original':
                data = self._apply_augmentation(data, aug_type)
                data = data.copy()
        
        data = torch.from_numpy(data).float()
        
        return data, basename

class LabeledDataset(Dataset):
    """Dataset class for labeled data (source domain)"""
    
    def __init__(self, data_ls, augmentation_types=None):
        self.data_ls = data_ls
        self.augmentation_types = augmentation_types
        
    def __len__(self):
        return len(self.data_ls)
    
    def _apply_augmentation(self, image, aug_type):
        # image is in (5, 41, 41)
        if aug_type.startswith('rotate_'):
            angle = int(aug_type.split('_')[1])
            k = int(angle / 90)
            image = np.rot90(image, k=k, axes=(1, 2))
            
        elif aug_type == 'flip_horizontal':
            image = np.flip(image, axis=2)
            
        elif aug_type == 'flip_vertical':
            image = np.flip(image, axis=1)
            
        elif aug_type == 'flip_diagonal':
            image = np.transpose(image, (0, 2, 1))
            
        elif aug_type == 'flip_anti-diagonal':
            image = np.flip(np.transpose(image, (0, 2, 1)), axis=2)
        
        return image
    
    def __getitem__(self, idx):
        filename = self.data_ls[idx]
        basename = os.path.basename(filename)
        label = 1 if '_sl_' in basename else 0
        data = np.load(filename)
        
        if self.augmentation_types is not None:
            aug_type = self.augmentation_types[idx]
            if aug_type != 'original':
                data = self._apply_augmentation(data, aug_type)
                data = data.copy()
        
        data = torch.from_numpy(data).float()
        label = torch.tensor(label)
        
        return data, label, basename

class BasicBlock(nn.Module):
    """Basic ResNet block for the custom ResNet model"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FeatureExtractor(nn.Module):
    """Feature extractor based on ResNet architecture"""
    
    def __init__(self, block, num_blocks):
        super(FeatureExtractor, self).__init__()
        self.in_planes = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        return out

class ClassClassifier(nn.Module):
    """Class classifier for binary classification"""
    
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super(ClassClassifier, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

class DomainClassifier(nn.Module):
    """Domain classifier for domain adaptation"""
    
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super(DomainClassifier, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

class DANNModelTransfer(nn.Module):
    
    def __init__(self, base_model, embedding_dim=512, hidden_dim=256):
        super().__init__()
        
        # Store the base model for reference
        self.base_model = base_model
        
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-3],
            nn.Flatten()
        )
        
        self.class_classifier = nn.Sequential(*list(base_model.children())[-3:])
        
        self.gradient_reversal = GradientReversalLayer()
        self.domain_classifier = DomainClassifier(embedding_dim=embedding_dim,
                                                  hidden_dim=hidden_dim)
        
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Class prediction
        class_output = self.class_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = self.gradient_reversal(features)
        domain_output = self.domain_classifier(reversed_features)
        
        return class_output, domain_output, features

def create_dann_dataloaders(train_source_ls, val_source_ls, blind_test_data_path, batch_size=256, 
                           save_path=None, seed=1234, augment_source=True, 
                           augment_target=False, source_subset_size=None):
    """
    Create dataloaders for DANN training
    """
    print('Creating DANN dataloaders')
    
    # Source data (labeled)
    train_source_ls = train_source_ls
    val_source_ls = val_source_ls
    
    # Blind test data (unlabeled)
    blind_test_ls = os.listdir(blind_test_data_path)
    blind_test_ls = [os.path.join(blind_test_data_path, file) for file in blind_test_ls]
    
    print('Source train dataset size:', len(train_source_ls))
    print('Source val dataset size:', len(val_source_ls))
    print('Blind test dataset size:', len(blind_test_ls))
    
    # Save data splits
    data_splits = {
        'num_train_source': len(train_source_ls),
        'num_val_source': len(val_source_ls),
        'num_blind_test': len(blind_test_ls),
        'train_source': train_source_ls,
        'val_source': val_source_ls,
        'blind_test': blind_test_ls
    }
    
    with open(os.path.join(save_path, 'data_splits_seed_{}.json'.format(seed)), 'w') as f:
        json.dump(data_splits, f, indent=4)
    
    # Create datasets
    val_source_dataset = LabeledDataset(val_source_ls)
    blind_test_dataset_unlabeled = BlindTestDataset(blind_test_ls)
    
    # Apply augmentation to training data
    augmentation_types = ['original', 'rotate_90', 'rotate_180', 'rotate_270',
                         'flip_horizontal', 'flip_vertical', 'flip_diagonal',
                         'flip_anti-diagonal']
    
    if augment_source:
        print('Using augmentation for source train dataset')
        train_source_ls_aug = []
        train_source_augmentation_types = []
        
        for filename in train_source_ls:
            for aug_type in augmentation_types:
                train_source_ls_aug.append(filename)
                train_source_augmentation_types.append(aug_type)
                
        train_source_dataset = LabeledDataset(train_source_ls_aug, train_source_augmentation_types)
    else:
        train_source_dataset = LabeledDataset(train_source_ls)
    
    if augment_target:
        print('Using augmentation for blind test dataset')
        blind_test_ls_aug = []
        blind_test_augmentation_types = []
        
        for filename in blind_test_ls:
            for aug_type in augmentation_types:
                blind_test_ls_aug.append(filename)
                blind_test_augmentation_types.append(aug_type)
                
        blind_test_dataset = BlindTestDataset(blind_test_ls_aug, blind_test_augmentation_types)
    else:
        blind_test_dataset = BlindTestDataset(blind_test_ls)
    
    print('Final source train dataset size:', len(train_source_dataset))
    print('Final blind test dataset size:', len(blind_test_dataset))
    
    # Create dataloaders
    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size, 
                                   shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    val_source_loader = DataLoader(val_source_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=12)
    blind_test_loader = DataLoader(blind_test_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    
    # Create a separate loader for final predictions (no shuffle, no drop_last)
    blind_test_pred_loader = DataLoader(blind_test_dataset_unlabeled, batch_size=batch_size, 
                                      shuffle=False, num_workers=12, drop_last=False)
    
    return train_source_loader, val_source_loader, blind_test_loader, blind_test_pred_loader

def train_dann_model(model, train_source_loader, val_source_loader, blind_test_loader,
                    epochs=100, lr=0.001, lambda_domain=1.0, device='cuda', save_path=None):
    """
    Train DANN model for domain adaptation
    """
    model.to(device)
    
    # Optimizers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    # Loss functions
    class_loss_fn = nn.BCEWithLogitsLoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()
    
    # History tracking
    train_loss_history = []
    val_loss_history = []
    component_loss_history = {
        'class_loss': [], 'domain_loss': [], 'total_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        
        model.train()
        
        # Calculate lambda for gradient reversal (increases over time)
        p = float(epoch) / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        model.gradient_reversal.alpha = alpha
        
        # Loss tracking
        train_class_loss_recorder = 0
        train_domain_loss_recorder = 0
        train_total_loss_recorder = 0
        train_data_counts = 0
        
        # Create iterators
        source_iter = iter(train_source_loader)
        target_iter = iter(blind_test_loader)
        
        # Determine number of batches
        n_batches = min(len(train_source_loader), len(blind_test_loader))
        
        for batch_idx in tqdm(range(n_batches), desc='Training'):
            # Get source batch
            try:
                source_data, source_labels, _ = next(source_iter)
            except StopIteration:
                source_iter = iter(train_source_loader)
                source_data, source_labels, _ = next(source_iter)
            
            # Get target batch
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(blind_test_loader)
                target_data, _ = next(target_iter)
            
            # Move to device
            source_data = source_data.to(device)
            source_labels = source_labels.to(device).float()
            target_data = target_data.to(device)
            
            batch_size = source_data.size(0)
            
            # Create domain labels (0 for source, 1 for target)
            source_domain_labels = torch.zeros(batch_size, 1).to(device)
            target_domain_labels = torch.ones(target_data.size(0), 1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass for source data
            source_class_output, source_domain_output, _ = model(source_data, alpha)
            
            # Forward pass for target data (only domain classification)
            _, target_domain_output, _ = model(target_data, alpha)
            
            # Calculate losses
            class_loss = class_loss_fn(source_class_output, source_labels.unsqueeze(1))
            
            domain_loss_source = domain_loss_fn(source_domain_output, source_domain_labels)
            domain_loss_target = domain_loss_fn(target_domain_output, target_domain_labels)
            domain_loss = domain_loss_source + domain_loss_target
            
            # Total loss
            total_loss = class_loss + lambda_domain * domain_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            train_class_loss_recorder += class_loss.item() * batch_size
            train_domain_loss_recorder += domain_loss.item() * batch_size
            train_total_loss_recorder += total_loss.item() * batch_size
            train_data_counts += batch_size
        
        # Calculate average losses
        avg_class_loss = train_class_loss_recorder / train_data_counts
        avg_domain_loss = train_domain_loss_recorder / train_data_counts
        avg_total_loss = train_total_loss_recorder / train_data_counts
        
        print('Train losses - Class: {:.4f}, Domain: {:.4f}, Total: {:.4f}, Alpha: {:.4f}'.format(
              avg_class_loss, avg_domain_loss, avg_total_loss, alpha))
        
        # Store losses
        train_loss_history.append(avg_total_loss)
        component_loss_history['class_loss'].append(avg_class_loss)
        component_loss_history['domain_loss'].append(avg_domain_loss)
        component_loss_history['total_loss'].append(avg_total_loss)
        
        # Validation
        model.eval()
        val_loss_recorder = 0
        val_correct = 0
        val_total = 0
        val_data_counts = 0
        
        with torch.no_grad():
            for source_data, source_labels, _ in tqdm(val_source_loader, desc='Validation'):
                source_data = source_data.to(device)
                source_labels = source_labels.to(device).float()
                
                batch_size = source_data.size(0)
                
                class_output, _, _ = model(source_data)
                loss = class_loss_fn(class_output, source_labels.unsqueeze(1))
                
                # Calculate accuracy
                predicted = (torch.sigmoid(class_output) > 0.5).float()
                val_correct += (predicted == source_labels.unsqueeze(1)).sum().item()
                val_total += source_labels.size(0)
                
                val_loss_recorder += loss.item() * batch_size
                val_data_counts += batch_size
        
        val_loss = val_loss_recorder / val_data_counts
        val_accuracy = val_correct / val_total
        
        print('Val - Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_accuracy))
        val_loss_history.append(val_loss)
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('New best model found! Saving checkpoint...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'component_loss_history': component_loss_history,
            }, os.path.join(save_path, 'dann_model_best.pth'))
        
        # Save training curves periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(train_loss_history, label='Train Loss')
            plt.plot(val_loss_history, label='Val Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(component_loss_history['class_loss'], label='Class Loss')
            plt.plot(component_loss_history['domain_loss'], label='Domain Loss')
            plt.title('Loss Components')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(component_loss_history['total_loss'], label='Total Loss')
            plt.title('Total Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'dann_training_history.png'))
            plt.close()

def predict_blind_test(model, model_path, blind_test_loader, device='cuda', save_path=None):
    """
    Make predictions on blind test data
    """
    # Load best model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    predictions = []
    probabilities = []
    filenames = []
    features_list = []
    
    print("Making predictions on blind test data...")
    
    with torch.no_grad():
        for data, names in tqdm(blind_test_loader, desc='Predicting'):
            data = data.to(device)
            
            class_output, _, features = model(data)
            probs = torch.sigmoid(class_output)
            preds = (probs > 0.5).float()
            
            predictions.append(preds.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            filenames.extend(names)
            features_list.append(features.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(predictions, axis=0).flatten()
    probabilities = np.concatenate(probabilities, axis=0).flatten()
    features_array = np.concatenate(features_list, axis=0)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'filename': filenames,
        'prediction': predictions.astype(int),
        'probability': probabilities
    })
    
    # Save results
    results_df.to_csv(os.path.join(save_path, 'blind_test_predictions.csv'), index=False)
    
    # Save features for analysis
    np.save(os.path.join(save_path, 'blind_test_features.npy'), features_array)
    
    print("Predictions saved to", os.path.join(save_path, 'blind_test_predictions.csv'))
    print("Features saved to", os.path.join(save_path, 'blind_test_features.npy'))
    
    return results_df, features_array

def analyze_predictions(results_df, features_array, save_path):
    """
    Analyze the predictions and create visualizations
    """
    print("Analyzing predictions...")
    
    # Basic statistics
    total_samples = len(results_df)
    positive_predictions = (results_df['prediction'] == 1).sum()
    negative_predictions = (results_df['prediction'] == 0).sum()
    
    print("Total samples:", total_samples)
    print("Predicted as SL (positive): {} ({:.1f}%)".format(positive_predictions, positive_predictions/total_samples*100))
    print("Predicted as Non-SL (negative): {} ({:.1f}%)".format(negative_predictions, negative_predictions/total_samples*100))
    
    # Probability distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(results_df['probability'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    prediction_counts = results_df['prediction'].value_counts()
    plt.bar(['Non-SL', 'SL'], [prediction_counts[0], prediction_counts[1]])
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    
    # t-SNE visualization of features
    plt.subplot(1, 3, 3)
    print("Computing t-SNE for feature visualization...")
    
    # Sample features for t-SNE if too many
    if len(features_array) > 10000:
        indices = np.random.choice(len(features_array), 10000, replace=False)
        features_sample = features_array[indices]
        predictions_sample = results_df['prediction'].iloc[indices].values
    else:
        features_sample = features_array
        predictions_sample = results_df['prediction'].values
    
    tsne = cuml.TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sample)
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=predictions_sample, cmap='viridis', alpha=0.6, s=1)
    plt.colorbar(scatter, label='Prediction (0=Non-SL, 1=SL)')
    plt.title('t-SNE of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis summary
    analysis_summary = {
        'total_samples': int(total_samples),
        'positive_predictions': int(positive_predictions),
        'negative_predictions': int(negative_predictions),
        'positive_percentage': float(positive_predictions/total_samples*100),
        'negative_percentage': float(negative_predictions/total_samples*100),
        'mean_probability': float(results_df['probability'].mean()),
        'std_probability': float(results_df['probability'].std()),
        'median_probability': float(results_df['probability'].median())
    }
    
    with open(os.path.join(save_path, 'analysis_summary.json'), 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print("Analysis complete! Results saved to", save_path)
    
    return analysis_summary

def prepare_submission(result_df, save_path):
    
    def get_id(filename):
        return filename.split('.')[0]
    
    submission_df = result_df[['filename', 'prediction']]
    submission_df.rename(columns={'filename': 'id', 'prediction': 'preds'}, inplace=True)
    submission_df['id'] = submission_df['id'].apply(get_id)

    submission_df.to_csv(os.path.join(save_path, 'submission_classification.csv'), index=False)

if __name__ == '__main__':
    # Configuration
    blind_test_data_path = '/home/xczhou/nis/xczhou/SL/LSST/datasets/numpy_files/test_dataset_updated_npy'
    
    source_path = '/home/xczhou/nis/xczhou/SL/LSST/classifier/results_deterministic_classifier_opt/'
    
    with open(os.path.join(source_path, 'data_splits_seed_42.json'), 'r') as f:
        data_splits = json.load(f)
    
    train_source_ls = data_splits['train']
    val_source_ls = data_splits['val']
    # Load pretrained model weights
    pretrained_model_path = os.path.join(source_path, 'best_deterministic_model.pth')
    
    # Training hyperparameters
    batch_size = 4096
    epochs = 100
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    lr = 0.0005
    lambda_domain = 1.0
    random_seed = 42
    augment_source = False
    augment_target = False
    source_subset_size = 'all'
    model_type = 'deterministic'
    
    # Create save directory
    save_path = './results_dann_adaptation_to_blind_test'
    os.makedirs(save_path, exist_ok=True)

    # Set up logging
    log_file = open(os.path.join(save_path, 'dann_logging.txt'), 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    print("=== DANN DOMAIN ADAPTATION TO BLIND TEST DATA ===")
    print("Source path:", source_path)
    print("Blind test data path:", blind_test_data_path)
    print("Pretrained model:", pretrained_model_path)
    print("Batch size:", batch_size)
    print("Epochs:", epochs)
    print("Learning rate:", lr)
    print("Lambda domain:", lambda_domain)
    print("Device:", device)
    print("Source subset size:", source_subset_size)
    print("Results will be saved to:", save_path)
    print()
    
    # Save configuration
    config = {
        'source_path': source_path,
        'blind_test_data_path': blind_test_data_path,
        'pretrained_model_path': pretrained_model_path,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': lr,
        'lambda_domain': lambda_domain,
        'random_seed': random_seed,
        'augment_source': augment_source,
        'augment_target': augment_target,
        'source_subset_size': source_subset_size,
        'model_type': model_type
    }
    
    with open(os.path.join(save_path, 'dann_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Load pretrained weights
    print("Loading pretrained weights...")
    pretrained_weights = torch.load(pretrained_model_path, map_location=device, weights_only=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_source_loader, val_source_loader, blind_test_loader, blind_test_pred_loader = \
        create_dann_dataloaders(train_source_ls, val_source_ls, blind_test_data_path, batch_size, 
                            save_path, random_seed, augment_source, augment_target, 
                            source_subset_size)
    
    # Create DANN model
    print("Initializing DANN model...")
    
    base_model = DeterministicResNetClassifier(BasicBlock, [2, 2, 2, 2])
    
    base_model.load_state_dict(pretrained_weights)
    model = DANNModelTransfer(base_model)
    model.to(device)
    
    # Train model
    print("Starting DANN training...")
    train_dann_model(model, train_source_loader, val_source_loader, blind_test_loader,
                    epochs, lr, lambda_domain, device, save_path)
        
    # Make predictions
    print("Making predictions on blind test data...")
    model_path = os.path.join(save_path, 'dann_model_best.pth')
    results_df, features_array = predict_blind_test(model, model_path, blind_test_pred_loader, 
                                                   device, save_path)
    
    # Analyze results
    print("Analyzing predictions...")
    analysis_summary = analyze_predictions(results_df, features_array, save_path)
    
    print("\n=== EXPERIMENT SUMMARY ===")
    print("Total blind test samples:", analysis_summary['total_samples'])
    print("Predicted as SL: {} ({:.1f}%)".format(analysis_summary['positive_predictions'], analysis_summary['positive_percentage']))
    print("Predicted as Non-SL: {} ({:.1f}%)".format(analysis_summary['negative_predictions'], analysis_summary['negative_percentage']))
    print("Mean probability: {:.4f}".format(analysis_summary['mean_probability']))
    print("Results saved to:", save_path)
    
    print('Preparing submission...')
    prepare_submission(results_df, save_path)
    
    print("DANN adaptation completed! Results saved to", save_path)
    
    # Close log file
    sys.stdout = original_stdout
    log_file.close()