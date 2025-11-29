"""
Dataset class for Mouse Behavior Detection

UPDATED to support:
1. Original keypoints (28 features)
2. Engineered features (50+ features with distances, velocities, angles)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter


class MouseBehaviorDataset(Dataset):
    """
    PyTorch Dataset for mouse behavior classification.
    
    Takes sequences of mouse keypoints/features and creates
    fixed-length windows for training.
    """
    
    def __init__(self, data, window_size=60, stride=10, balance_classes=True, use_features=True):
        """
        Initialize dataset.
        
        Args:
            data: Dict with 'sequences' and 'vocabulary'
            window_size: Number of frames per training example
            stride: Step between consecutive windows
            balance_classes: Whether to compute class weights for balanced sampling
            use_features: If True, use engineered features. If False, use raw keypoints.
        """
        self.window_size = window_size
        self.stride = stride
        self.vocabulary = data['vocabulary']
        self.num_classes = len(self.vocabulary)
        self.use_features = use_features
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.vocabulary)}
        
        print(f"\n📦 Creating dataset...")
        print(f"   Window size: {window_size} frames")
        print(f"   Stride: {stride} frames")
        print(f"   Classes: {self.num_classes}")
        print(f"   Using features: {use_features}")
        
        # Extract windows from all sequences
        self.samples = []
        self.labels = []
        
        for seq_id, seq_data in data['sequences'].items():
            # Choose data source based on use_features flag
            if use_features and 'features' in seq_data:
                # Use engineered features (already normalized)
                features = seq_data['features']  # Shape: (num_frames, num_features)
            else:
                # Use raw keypoints (flatten to 2D)
                keypoints = seq_data['keypoints']  # Shape: (frames, mice, coords, bodyparts)
                num_frames = keypoints.shape[0]
                features = keypoints.reshape(num_frames, -1)  # Flatten
            
            annotations = seq_data['annotations']
            num_frames = len(annotations)
            
            # Create windows
            for start in range(0, num_frames - window_size + 1, stride):
                end = start + window_size
                
                # Get window data
                window = features[start:end]
                
                # Get label for center frame (better than first or last)
                center_frame = start + window_size // 2
                label = annotations[center_frame]
                label_idx = self.label_to_idx.get(label, 0)  # Default to 'other'
                
                self.samples.append(window)
                self.labels.append(label_idx)
        
        # Convert to numpy arrays
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"\n   Created {len(self.samples):,} training windows")
        print(f"   Sample shape: {self.samples.shape}")
        
        # Compute class weights for balanced sampling
        self.sample_weights = None
        if balance_classes:
            self._compute_class_weights()
    
    def _compute_class_weights(self):
        """
        Compute sample weights for balanced sampling.
        
        Rare classes get higher weights so they're sampled more often.
        """
        label_counts = Counter(self.labels)
        total = len(self.labels)
        
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total / (count * self.num_classes)
        
        self.sample_weights = np.array([weights[label] for label in self.labels])
        
        print(f"\n   Class weights (higher = more important):")
        for label in sorted(weights.keys()):
            behavior = self.vocabulary[label]
            count = label_counts[label]
            weight = weights[label]
            print(f"     {behavior:20s}: {count:7,} samples, weight: {weight:.3f}")
    
    def get_sampler(self):
        """Create weighted sampler for balanced batches."""
        if self.sample_weights is not None:
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get one training example."""
        sample = torch.FloatTensor(self.samples[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sample, label
    
    def get_class_distribution(self):
        """Get distribution of classes."""
        return Counter(self.labels)


class MouseBehaviorDatasetWithAugmentation(MouseBehaviorDataset):
    """
    Extended dataset with data augmentation.
    
    Augmentations help the model generalize better, especially
    for rare behaviors like 'approach' and 'genitalgroom'.
    """
    
    def __init__(self, data, window_size=60, stride=10, balance_classes=True, 
                 use_features=True, augment=True):
        super().__init__(data, window_size, stride, balance_classes, use_features)
        
        self.augment = augment
        self.training = True
        
        if augment:
            print(f"\n   Data augmentation ENABLED")
    
    def set_training(self, mode=True):
        """Set whether to apply augmentation (only during training)."""
        self.training = mode
    
    def __getitem__(self, idx):
        """Get one training example with optional augmentation."""
        sample = self.samples[idx].copy()  # Copy to avoid modifying original
        label = self.labels[idx]
        
        if self.augment and self.training:
            sample = self._augment(sample)
        
        sample = torch.FloatTensor(sample)
        label = torch.LongTensor([label])[0]
        return sample, label
    
    def _augment(self, sample):
        """
        Apply augmentation to a sample.
        
        Augmentations:
        1. Gaussian noise (simulates tracking noise)
        2. Time shift (random crop within window)
        3. Scaling (size variation)
        """
        # 1. Add small Gaussian noise (50% chance)
        if np.random.random() < 0.5:
            noise_scale = 0.02 * np.std(sample)
            noise = np.random.normal(0, noise_scale, sample.shape)
            sample = sample + noise.astype(np.float32)
        
        # 2. Random temporal jitter (shift frames slightly, 30% chance)
        if np.random.random() < 0.3:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                sample = np.roll(sample, shift, axis=0)
        
        # 3. Random scaling (simulate different sizes/distances, 30% chance)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.95, 1.05)
            sample = sample * scale
        
        return sample


# ============================================
# QUICK TEST
# ============================================
if __name__ == "__main__":
    import pickle
    from pathlib import Path
    
    print("Testing dataset...")
    
    # Try to load real data
    data_path = Path('/Users/tasi/Desktop/MABE/data/processed/train_calms21_task1.pkl')
    
    if data_path.exists():
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        dataset = MouseBehaviorDataset(
            data,
            window_size=60,
            stride=10,
            balance_classes=True,
            use_features=False  # Original data doesn't have features
        )
        
        print(f"\n✅ Dataset test passed!")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Sample shape: {dataset.samples.shape}")
        
        # Test getting an item
        sample, label = dataset[0]
        print(f"   Sample tensor shape: {sample.shape}")
        print(f"   Label: {label.item()} ({dataset.vocabulary[label.item()]})")
    else:
        print("   Data file not found, skipping real data test")
        
        # Create dummy data for testing
        dummy_data = {
            'vocabulary': ['other', 'approach', 'attack', 'mount'],
            'sequences': {
                'test_seq': {
                    'keypoints': np.random.randn(1000, 2, 2, 7).astype(np.float32),
                    'annotations': ['other'] * 500 + ['approach'] * 200 + ['attack'] * 200 + ['mount'] * 100
                }
            }
        }
        
        dataset = MouseBehaviorDataset(
            dummy_data,
            window_size=60,
            stride=10,
            balance_classes=True,
            use_features=False
        )
        
        print(f"\n✅ Dataset test with dummy data passed!")
        print(f"   Total samples: {len(dataset)}")
