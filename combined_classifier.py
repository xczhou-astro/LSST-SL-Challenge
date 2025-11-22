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
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report


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
        
class DeterministicResNetClassifier(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
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
        
        # Feature extraction output
        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.batch = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batch(x)
        x = self.fc2(x)
        return x
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty estimation
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            pred = torch.sigmoid(logits)
            return pred
        
def create_dataloaders(train_hsc_ls, train_slsim_ls, 
                       val_hsc_ls, val_slsim_ls, 
                       test_hsc_ls, test_slsim_ls, 
                       batch_size=256, save_path=None, 
                       seed=1234, augment=True):
    
    train_all_ls = train_hsc_ls + train_slsim_ls
    val_all_ls = val_hsc_ls + val_slsim_ls
    
    data_splits = {
        'num_train': len(train_all_ls),
        'num_val': len(val_all_ls),
        'num_test_hsc': len(test_hsc_ls),
        'num_test_slsim': len(test_slsim_ls),
        'num_hsc_train': len(train_hsc_ls),
        'num_slsim_train': len(train_slsim_ls),
        'num_hsc_val': len(val_hsc_ls),
        'num_slsim_val': len(val_slsim_ls),
        'num_hsc_test': len(test_hsc_ls),
        'num_slsim_test': len(test_slsim_ls),
        'train': train_all_ls,
        'val': val_all_ls,
        'test_hsc': test_hsc_ls,
        'test_slsim': test_slsim_ls
    }
    
    with open(os.path.join(save_path, 'data_splits_seed_{}.json'.format(seed)), 'w') as f:
        json.dump(data_splits, f, indent=4)
    
    val_all_dataset = LabeledDataset(val_all_ls)
    
    test_hsc_dataset = LabeledDataset(test_hsc_ls)
    test_slsim_dataset = LabeledDataset(test_slsim_ls)
    
    # Apply augmentation to training data
    augmentation_types = ['original', 'rotate_90', 'rotate_180', 'rotate_270',
                         'flip_horizontal', 'flip_vertical', 'flip_diagonal',
                         'flip_anti-diagonal']
    
    if augment:
        print('Using augmentation for train dataset')
        train_all_ls_aug = []
        train_all_augmentation_types = []
        
        for filename in train_all_ls:
            for aug_type in augmentation_types:
                train_all_ls_aug.append(filename)
                train_all_augmentation_types.append(aug_type)
                
        train_all_dataset = LabeledDataset(train_all_ls_aug, train_all_augmentation_types)
    else:
        train_all_dataset = LabeledDataset(train_all_ls)
    
    print('Final train dataset size:', len(train_all_dataset))
    print('Final val dataset size:', len(val_all_dataset))
    print('Final test hsc dataset size:', len(test_hsc_dataset))
    print('Final test slsim dataset size:', len(test_slsim_dataset))
    
    train_all_loader = DataLoader(train_all_dataset, batch_size=batch_size, 
                                   shuffle=True, num_workers=12)
    val_all_loader = DataLoader(val_all_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=12)
    test_hsc_loader = DataLoader(test_hsc_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=12)
    test_slsim_loader = DataLoader(test_slsim_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=12)
        
    return train_all_loader, val_all_loader, test_hsc_loader, test_slsim_loader

def train_model(model, train_all_loader, val_all_loader, epochs=100, lr=0.001, 
                        device='cuda', save_path=None, model_type='deterministic'):
    """
    Train model with appropriate loss function
    """
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    
    if model_type == 'bayesian':
        # Use ELBO loss for Bayesian model
        def loss_fn(outputs, targets):
            return model.elbo_loss(outputs, targets)
    else:
        # Use binary cross-entropy for MC Dropout
        criterion = nn.BCEWithLogitsLoss()
        def loss_fn(outputs, targets):
            loss = criterion(outputs.squeeze(), targets.float())
            return loss, loss, torch.tensor(0.0)
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs}')
        
        # Training
        model.train()
        train_batch_loss = 0
        train_batch_count = 0
        
        for batch_idx, (images, labels, _) in enumerate(tqdm(train_all_loader, desc='Training')):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss, nll, kl = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_batch_loss += loss.item()
            train_batch_count += 1
        
        train_loss_history.append(train_batch_loss / train_batch_count)
        print(f'Epoch {epoch}/{epochs}, Loss: {train_loss_history[-1]}')
        
        # Validation
        model.eval()
        val_batch_loss = 0
        val_batch_count = 0
        
        for batch_idx, (images, labels, _) in enumerate(tqdm(val_all_loader, desc='Validating')):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(images)
                loss, nll, kl = loss_fn(outputs, labels)
            
            val_batch_loss += loss.item()
            val_batch_count += 1
        
        val_loss_history.append(val_batch_loss / val_batch_count)
        print(f'Epoch {epoch}/{epochs}, Val Loss: {val_loss_history[-1]}')
        
        if val_loss_history[-1] < best_val_loss:
            best_val_loss = val_loss_history[-1]
            torch.save(model.state_dict(), os.path.join(save_path, f'best_{model_type}_model.pth'))
            
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr:.6f} -> {new_lr:.6f}')
    
    # Save training history
    df_loss = pd.DataFrame({'train_loss': train_loss_history, 'val_loss': val_loss_history})
    df_loss.to_csv(os.path.join(save_path, f'{model_type}_loss.csv'), index=False)
    
    plot_history(train_loss_history, val_loss_history, save_path)
    
    return train_loss_history, val_loss_history

def plot_history(train_loss_history, val_loss_history, save_path):

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses.png'))
    plt.close()

def evaluate_deterministic(model, weights_path, data_loader, device='cuda'):
    
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    all_predictions = []
    all_labels = []
    all_filenames = []

    for batch_idx, (images, labels, filenames) in enumerate(tqdm(data_loader, desc='Testing with Deterministic')):
        images = images.to(device)
        
        with torch.no_grad():
            pred = model.predict_with_uncertainty(images)
        
        all_predictions.append(pred.cpu().numpy())
        all_labels.append(labels.numpy())
        all_filenames.append(filenames)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_filenames = np.concatenate(all_filenames, axis=0)
    
    return all_predictions, all_labels, all_filenames

def plot_cm(probs, labels, threshold, ds, save_path):
    """
    Plot confusion matrix and classification metrics
    
    Args:
        probs: Probability scores (before thresholding)
        labels: Ground truth labels
        threshold: Classification threshold
        ds: Dataset name (for saving)
        save_path: Directory to save results
    """
    probs = probs.reshape(-1)
    labels = labels.reshape(-1).astype(int)
    
    predictions = (probs > threshold).astype(int)
    
    print(f'=== {ds.upper()} DOMAIN EVALUATION ===')
    print(f'Threshold: {threshold:.2f}')
    
    print(f'Class distribution:')
    print(f'  - SL (positive): {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)')
    print(f'  - Non-SL (negative): {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)')
    
    print(f'Predictions:')
    print(f'  - Positive: {np.sum(predictions == 1)} ({np.mean(predictions == 1)*100:.1f}%)')
    print(f'  - Negative: {np.sum(predictions == 0)} ({np.mean(predictions == 0)*100:.1f}%)')
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Calculate metrics
    report = classification_report(labels, predictions, 
                                  target_names=['Non-SL', 'SL'],
                                  output_dict=True)
    
    f1 = f1_score(labels, predictions)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, 
                              target_names=['Non-SL', 'SL']))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels and title
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Non-SL', 'SL'],
           yticklabels=['Non-SL', 'SL'],
           title=f'Confusion Matrix - {ds.upper()} F1={f1:.2f}',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations with percentages
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Show count and percentage
            count = cm[i, j]
            percentage = 100 * count / np.sum(cm[i, :])
            ax.text(j, i, f"{count}\n({percentage:.1f}%)",
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{ds}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'accuracy': report['accuracy'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score'],
        'sl_precision': report['SL']['precision'],
        'sl_recall': report['SL']['recall'],
        'sl_f1': report['SL']['f1-score'],
        'non_sl_precision': report['Non-SL']['precision'],
        'non_sl_recall': report['Non-SL']['recall'],
        'non_sl_f1': report['Non-SL']['f1-score'],
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(save_path, f'metrics_{ds}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == '__main__':
    
    hsc_data_path = '../datasets/numpy_files/hsc'
    slsim_data_path = '../datasets/numpy_files/slsim'
    
    hsc_data_path = np.array([os.path.join(hsc_data_path, f) for f in os.listdir(hsc_data_path)])
    slsim_data_path = np.array([os.path.join(slsim_data_path, f) for f in os.listdir(slsim_data_path)])
    
    val_split = 0.1
    test_split = 0.1
    
    np.random.seed(42)
    hsc_size = len(hsc_data_path)
    val_hsc_idx = np.random.choice(hsc_size, size=int(hsc_size * val_split), replace=False)
    train_hsc_idx = np.setdiff1d(np.arange(hsc_size), val_hsc_idx)
    np.random.seed(42)
    test_hsc_idx = np.random.choice(train_hsc_idx, size=int(hsc_size * test_split), replace=False)
    train_hsc_idx = np.setdiff1d(train_hsc_idx, test_hsc_idx)
    
    slsim_size = len(slsim_data_path)
    np.random.seed(42)
    val_slsim_idx = np.random.choice(slsim_size, size=int(slsim_size * val_split), replace=False)
    train_slsim_idx = np.setdiff1d(np.arange(slsim_size), val_slsim_idx)
    np.random.seed(42)
    test_slsim_idx = np.random.choice(train_slsim_idx, size=int(slsim_size * test_split), replace=False)
    train_slsim_idx = np.setdiff1d(train_slsim_idx, test_slsim_idx)
    
    hsc_val_data_ls = hsc_data_path[val_hsc_idx].tolist()
    slsim_val_data_ls = slsim_data_path[val_slsim_idx].tolist()
    
    hsc_test_data_ls = hsc_data_path[test_hsc_idx].tolist()
    slsim_test_data_ls = slsim_data_path[test_slsim_idx].tolist()
    
    hsc_train_data_ls = hsc_data_path[train_hsc_idx].tolist()
    slsim_train_data_ls = slsim_data_path[train_slsim_idx].tolist()
    
    batch_size = 8192
    epochs = 60
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    lr = 0.0002
    random_seed = 42
    augment = True
    model_type = 'deterministic'
    
    save_path = f'results_{model_type}_classifier'
    os.makedirs(save_path, exist_ok=True)
    
    log_file = open(os.path.join(save_path, 'logging.txt'), 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    train_hsc_ls = hsc_train_data_ls
    train_slsim_ls = slsim_train_data_ls
    
    val_hsc_ls = hsc_val_data_ls
    val_slsim_ls = slsim_val_data_ls
    
    test_hsc_ls = hsc_test_data_ls
    test_slsim_ls = slsim_test_data_ls
    
    train_all_loader, val_all_loader, test_hsc_loader, test_slsim_loader = \
        create_dataloaders(
            train_hsc_ls, 
            train_slsim_ls,
            val_hsc_ls,
            val_slsim_ls,
            test_hsc_ls,
            test_slsim_ls,
            batch_size=batch_size,
            save_path=save_path,
            seed=random_seed,
            augment=augment
        )
    
    model = DeterministicResNetClassifier(BasicBlock, [2, 2, 2, 2])
    
    train_loss_history, val_loss_history = train_model(
        model,
        train_all_loader,
        val_all_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path=save_path,
        model_type=model_type
    )
    
    weights_path = os.path.join(save_path, 'best_{}_model.pth'.format(model_type))
    
    hsc_predictions, hsc_labels, hsc_filenames = \
        evaluate_deterministic(
            model,
            weights_path,
            test_hsc_loader, 
            device=device)
        
    slsim_predictions, slsim_labels, slsim_filenames = \
        evaluate_deterministic(
            model,
            weights_path,
            test_slsim_loader,
            device=device)
    
    plot_cm(hsc_predictions, hsc_labels, 0.5, 'hsc', save_path)
    plot_cm(slsim_predictions, slsim_labels, 0.5, 'slsim', save_path)
    
    all_predictions = np.concatenate([hsc_predictions, slsim_predictions], axis=0)
    all_labels = np.concatenate([hsc_labels, slsim_labels], axis=0)
    all_filenames = np.concatenate([hsc_filenames, slsim_filenames], axis=0)

    ds_names = ['hsc'] * len(hsc_labels) + ['slsim'] * len(slsim_labels)
    
    plot_cm(all_predictions, all_labels, 0.5, 'all', save_path)
    
    df = pd.DataFrame({
        'filename': all_filenames.reshape(-1),
        'pred_prob_sl': all_predictions.reshape(-1),
        'true_label': all_labels.reshape(-1),
        'ds_name': ds_names
    })
    
    df.to_csv(os.path.join(save_path, 'results.csv'), index=False)

    sys.stdout = original_stdout
    log_file.close()