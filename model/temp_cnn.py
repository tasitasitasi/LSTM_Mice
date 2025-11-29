import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    """
    Convolutional Neural Network for time-series data
    """

    def __init__(self, input_dim=28, num_classes=10, window_size=30, dropout=0.5):
        """
        Build the neural network

        Args:
            input_dim: Number of features per frame
            num_classes: Number of behaviors to predict
            window_size: Number of frames in each input
            dropout: Dropout rate
        """
        super(TemporalCNN, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        print(f"\n🧠 Building neural network...")
        print(f"   Input: {input_dim} features per frame")
        print(f"   Output: {num_classes} behavior classes")

        # Layer 1
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        # Layer 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        # Layer 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor (batch_size, window_size, input_dim)

        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Reshape for Conv1d
        x = x.permute(0, 2, 1)

        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Global pooling
        x = self.gap(x)
        x = x.squeeze(-1)

        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return
