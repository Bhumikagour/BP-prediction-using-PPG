"""
PyTorch Deep Learning Model Architecture Module.
Implements a 1D-ResNet Backbone + Bidirectional LSTM Recurrent Network
for 3-Channel Waveform Blood Pressure Regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock1D(nn.Module):
    """
    1D Residual Block with 2 Conv1D layers, BatchNorm, LeakyReLU, and Skip Connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        # Shortcut connection for dimension matching
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.act2(out)

class PPGResNetBiLSTM(nn.Module):
    """
    Hybrid 1D-ResNet + Bidirectional LSTM Architecture for Continuous BP Estimation.
    Input Shape: [batch_size, 3, 1500] (PPG, VPG, APG)
    Output Shape: [batch_size, 3] (SBP, DBP, MAP in mmHg)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_outputs: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.25
    ):
        super(PPGResNetBiLSTM, self).__init__()

        # Initial Stem Convolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Layers
        self.layer1 = ResNetBlock1D(32, 64, stride=2)
        self.layer2 = ResNetBlock1D(64, 128, stride=2)
        self.layer3 = ResNetBlock1D(128, 256, stride=2)

        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Regression Head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 3, 1500]
        out = self.stem(x)         # [B, 32, 375]
        out = self.layer1(out)     # [B, 64, 188]
        out = self.layer2(out)     # [B, 128, 94]
        out = self.layer3(out)     # [B, 256, 47]

        # Permute for LSTM: [B, C, L] -> [B, L, C]
        out = out.permute(0, 2, 1) # [B, 47, 256]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(out) # [B, 47, 2*lstm_hidden]
        
        # Take global average pooling over sequence dimension
        pooled = torch.mean(lstm_out, dim=1) # [B, 2*lstm_hidden]

        # Predict SBP, DBP, MAP
        preds = self.fc(pooled)     # [B, 3]
        return preds
