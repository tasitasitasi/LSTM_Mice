"""
Training script for CNN-LSTM model with engineered features
--Uses engineered features (distances, velocities, angles)
--LSTM architecture for better temporal modeling
--Data augmentation for rare classes
--handling of class imbalance
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
from collections import Counter
import time

# Import models - try CNN-LSTM first, fall back to original CNN
try:
    from models.temporal_cnn_lstm import TemporalCNNLSTM, TemporalCNN
    print("Using CNN-LSTM model")
    USE_LSTM = True
except ImportError:
    from models.temporal_cnn import TemporalCNN
    print("CNN-LSTM not found, using original CNN")
    USE_LSTM = False

from data.dataset import MouseBehaviorDataset


class FocalLoss(nn.Module):
    """
    Focal Loss - helps with class imbalance
    
    Focuses training on hard examples (confused predictions).
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            reduction='none',
            weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class MixupAugmentation:
    """
    Mixup data augmentation - helps with generalization.
    
    Creates synthetic training examples by blending two real examples together.
    Especially helpful for rare classes!
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y, device):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup - blend loss from both original examples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    mixup = MixupAugmentation(alpha=mixup_alpha) if use_mixup else None

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup and np.random.random() < 0.5:  # Apply mixup 50% of the time
            mixed_data, y_a, y_b, lam = mixup(data, target, device)
            output = model(mixed_data)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            
            # For tracking, use original targets
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
        else:
            output = model(data)
            loss = criterion(output, target)
            
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (important for LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'    Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1


def validate(model, loader, criterion, device, vocabulary):
    """Evaluate on validation data."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    return avg_loss, f1, all_preds, all_labels, f1_per_class


def main():
    # ===== CONFIGURATION =====
    print("=" * 70)
    print("MABE MOUSE BEHAVIOR - CNN-LSTM TRAINING")
    print("=" * 70)

    # Hyperparameters
    WINDOW_SIZE = 60          # Frames per window
    STRIDE = 10               # Overlap between windows
    BATCH_SIZE = 64           # Smaller batch for LSTM (uses more memory)
    NUM_EPOCHS = 100          # More epochs for complex model
    LEARNING_RATE = 0.0005    # Lower LR for LSTM stability
    DROPOUT = 0.5             # Regularization
    LSTM_HIDDEN = 128         # LSTM hidden size
    LSTM_LAYERS = 2           # Number of LSTM layers
    USE_MIXUP = True          # Data augmentation
    
    print(f"\n⚙️  Configuration:")
    print(f"   Window size: {WINDOW_SIZE} frames")
    print(f"   Stride: {STRIDE} frames")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   LSTM hidden: {LSTM_HIDDEN}")
    print(f"   LSTM layers: {LSTM_LAYERS}")
    print(f"   Use Mixup: {USE_MIXUP}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 Using device: {device}")
    if device.type == 'cpu':
        print("   Note: Training on CPU. LSTM is slower than CNN.")
        print("   Estimated time: 4-6 hours")

    # load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

 
    original_path = project_root / 'data/processed/train_calms21_task1.pkl'
    
    with open(data_path, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loaded!")
    print(f"   Sequences: {len(train_data['sequences'])}")
    print(f"   Behaviors: {train_data['vocabulary']}")
    
    if use_features and 'feature_names' in train_data:
        print(f"   Features per frame: {len(train_data['feature_names'])}")

    print("\n" + "=" * 70)
    print("CREATING DATASET")
    print("=" * 70)

    full_dataset = MouseBehaviorDataset(
        train_data,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        balance_classes=True,
        use_features=use_features  # NEW: use engineered features if available
    )

    # train val split
    print(f"\n Splitting into train/validation sets")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train: {len(train_dataset):,} windows")
    print(f"   Validation: {len(val_dataset):,} windows")

    # load data

    train_indices = train_dataset.indices
    train_sampler_weights = full_dataset.sample_weights[train_indices]

    train_sampler = WeightedRandomSampler(
        weights=train_sampler_weights,
        num_samples=len(train_sampler_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"   Train batches per epoch: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    # model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    input_dim = full_dataset.samples.shape[2]
    num_classes = len(train_data['vocabulary'])

    if USE_LSTM:
        model = TemporalCNNLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            window_size=WINDOW_SIZE,
            dropout=DROPOUT,
            lstm_hidden=LSTM_HIDDEN,
            lstm_layers=LSTM_LAYERS,
            bidirectional=True
        )
    else:
        model = TemporalCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            window_size=WINDOW_SIZE,
            dropout=DROPOUT
        )
    
    model = model.to(device)

    # loss function
    print(f"\nCreating loss function with class weights...")

    class_counts = [0] * num_classes
    for label in full_dataset.labels[train_indices]:
        class_counts[label] += 1

    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = len(train_indices) / (count * num_classes)
        else:
            weight = 0
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"   Class weights:")
    for idx, (behavior, weight) in enumerate(zip(train_data['vocabulary'], class_weights)):
        print(f"     {behavior:20s}: {weight:.2f}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Cosine annealing scheduler 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,      # Restart every 20 epochs
        T_mult=2,    # Double the period after each restart
        eta_min=1e-6
    )

    # training loop

    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 20

    start_time = time.time()

    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            
            print(f"\n{'=' * 70}")
            print(f"EPOCH {epoch + 1}/{NUM_EPOCHS} (LR: {scheduler.get_last_lr()[0]:.6f})")
            print(f"{'=' * 70}")

            # Train
            print(f"\n🏋️  Training...")
            train_loss, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device,
                use_mixup=USE_MIXUP, mixup_alpha=0.2
            )

            # Validate
            print(f"\nValidating...")
            val_loss, val_f1, val_preds, val_labels, f1_per_class = validate(
                model, val_loader, criterion, device, train_data['vocabulary']
            )

            # Update scheduler
            scheduler.step()

            # Timing
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Results
            print(f"\nResults:")
            print(f"   Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"   Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
            print(f"   Epoch time: {epoch_time/60:.1f} min | Total: {total_time/3600:.1f} hrs")

            # Per-class F1
            print(f"\n   Per-class F1 scores:")
            for idx, (behavior, f1) in enumerate(zip(train_data['vocabulary'], f1_per_class)):
                # Add indicator for improvement target classes
                indicator = "🎯" if behavior in ['approach', 'genitalgroom', 'attack'] else ""
                print(f"     {behavior:20s}: {f1:.3f} {indicator}")

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0

                model_path = project_root / 'models/cnn_lstm_best.pth'
                model_path.parent.mkdir(exist_ok=True, parents=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'f1_per_class': f1_per_class,
                    'vocabulary': train_data['vocabulary'],
                    'input_dim': input_dim,
                    'model_type': 'CNN-LSTM' if USE_LSTM else 'CNN'
                }, model_path)


    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # end
    total_time = time.time() - start_time
    

    print(f"\n Best validation F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f" Total training time: {total_time/3600:.1f} hours")
    print(f" Best model saved to: models/cnn_lstm_best.pth")
    
    # Load best model and show final results
    checkpoint = torch.load(project_root / 'models/cnn_lstm_best.pth')
    print(f"\n📊 Final per-class F1 scores (best model):")
    for behavior, f1 in zip(checkpoint['vocabulary'], checkpoint['f1_per_class']):
        status = "good if f1 >= 0.5 else "medium" if f1 >= 0.3 else "bad"
        print(f"   {behavior:20s}: {f1:.3f} {status}")


if __name__ == "__main__":
    main()
