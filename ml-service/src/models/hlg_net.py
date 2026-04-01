"""
HLG-Net: Hierarchical Local-Global Network for depression detection.

A lightweight CNN + Sigmoid Multi-Head Attention architecture that processes
40-D MFCC features to predict continuous depression severity scores.
Only 0.049M parameters — suitable for edge deployment.

Architecture:
    Audio → MFCC (40-D, 4687 frames) → 3× Conv1D blocks → Sigmoid MHA → AvgPool → Linear → Score

Reference: PLAN.md (HLG-Net replication guide)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single Conv1D + MaxPool1D block for local feature extraction.

    Each block captures hierarchical acoustic patterns:
      - Layer 1: Low-level (fundamental frequency, energy)
      - Layer 2: Mid-level (formant blurring, clarity)
      - Layer 3: High-level (composite articulation features)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool_size: int = 5, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SigmoidMultiHeadAttention(nn.Module):
    """Sigmoid-based Multi-Head Attention for global feature extraction.

    Uses sigmoid instead of softmax for attention weights, allowing
    multi-point activation. This better captures distributed acoustic
    anomalies in depressed speech (vs. softmax's unimodal focus).

    Attention(Q, K, V) = Sigmoid((Q·K^T) / √d_k) · V
    """

    def __init__(self, d_model: int = 64, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections → split into heads
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Sigmoid attention (NOT softmax — key difference from standard transformers)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.sigmoid(scores)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Final linear projection
        return self.W_O(attn_output)


class HLGNet(nn.Module):
    """Hierarchical Local-Global Network for depression severity prediction.

    Processes 40-D MFCC features through:
      1. Local feature extraction: 3 stacked Conv1D blocks
      2. Global feature extraction: Sigmoid multi-head attention
      3. Classification: Average pooling → linear projection

    Args:
        input_dim: Number of MFCC coefficients (default: 40)
        d_model: Hidden dimension for conv blocks and attention (default: 64)
        num_heads: Number of attention heads (default: 8)

    Input shape:  (batch, seq_len=4687, input_dim=40)
    Output shape: (batch,) — continuous depression score
    """

    def __init__(self, input_dim: int = 40, d_model: int = 64, num_heads: int = 8):
        super().__init__()

        # Local Feature Extraction (3 Conv blocks)
        self.conv1 = ConvBlock(input_dim, d_model, kernel_size=3, pool_size=5)
        self.conv2 = ConvBlock(d_model, d_model, kernel_size=3, pool_size=5)
        self.conv3 = ConvBlock(d_model, d_model, kernel_size=3, pool_size=5)

        # Global Feature Extraction (Sigmoid-MHA)
        self.mha = SigmoidMultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Classification
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len=4687, input_dim=40)
        Returns:
            Depression score of shape (batch,)
        """
        # Transpose for Conv1D: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Local feature extraction
        x = self.conv1(x)  # → (batch, 64, 937)
        x = self.conv2(x)  # → (batch, 64, 187)
        x = self.conv3(x)  # → (batch, 64, 37)

        # Transpose back for attention: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # → (batch, 37, 64)

        # Global feature extraction
        x = self.mha(x)  # → (batch, 37, 64)

        # Classification
        x = x.transpose(1, 2)  # → (batch, 64, 37)
        x = self.avgpool(x)    # → (batch, 64, 1)
        x = x.squeeze(-1)      # → (batch, 64)
        x = self.dropout(x)
        x = self.fc(x)         # → (batch, 1)

        return x.squeeze(-1)   # → (batch,)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
