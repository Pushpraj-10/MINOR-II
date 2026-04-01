HLG-Net: Speech-Based Depression Recognition
A hierarchical local-global network for automatic depression detection from speech using CNNs and sigmoid-based multi-head attention.
Overview
HLG-Net achieves MAE 5.13 on AVEC 2014 with only 0.049M parameters and 0.021G FLOPs, making it suitable for edge deployment. The model captures both local acoustic anomalies (monotonic pitch, weakened formants) and global utterance-level degradation (emotional attenuation, semantic fragmentation).

Architecture
Audio Input → MFCC (40-D) → Local Feature Extraction → Global Feature Extraction → Classification → BDI-II Score
1. Local Feature Extraction
Three stacked convolutional blocks:

Conv1D Block 1: Conv1D(40→64, k=3) → MaxPool1D(k=5, s=5)
Conv1D Block 2: Conv1D(64→64, k=3) → MaxPool1D(k=5, s=5)
Conv1D Block 3: Conv1D(64→64, k=3) → MaxPool1D(k=5, s=5)

Each block extracts hierarchical patterns:

Layer 1: Low-level (fundamental frequency, energy)
Layer 2: Mid-level (formant blurring, clarity)
Layer 3: High-level (composite articulation features)

2. Global Feature Extraction
Sigmoid-based Multi-Head Attention (8 heads):
python# Attention computation
Q, K, V = x³_c @ W_Q, x³_c @ W_K, x³_c @ W_V
Attention(Q, K, V) = Sigmoid((Q·K^T) / √d_k) · V
MHA(x) = Concat(Head_1, ..., Head_8) @ W_O
Key advantage: Sigmoid allows multi-point activation (vs. softmax's unimodal focus), better capturing distributed acoustic anomalies in depressed speech.
3. Classification

Average pooling (37→1)
Linear projection (64→1)
Output: Continuous BDI-II score [0-63]


Step-by-Step Replication Guide
Prerequisites
bash# Environment
Python 3.8+
PyTorch 1.12+
torchaudio
librosa
numpy
scikit-learn
MQBench  # For quantization
Step 1: Data Preparation
Dataset: AVEC 2014 (link)

300 audio clips (structured + free expression tasks)
BDI-II scores [0-63] annotated per clip
Duration: 6s to 3min

MFCC Extraction:
pythonimport librosa

def extract_mfcc(audio_path, n_mfcc=40, frame_size=4687):
    """Extract 40-D MFCC features"""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # MFCC with 100 Hz frame rate (10ms hop)
    mfcc = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=n_mfcc,
        n_fft=400,      # 25ms window at 16kHz
        hop_length=160  # 10ms hop → 100 Hz frame rate
    )
    
    # Standardize to 4687 frames (valid convolution mode)
    if mfcc.shape[1] < frame_size:
        # Pad shorter sequences
        pad_width = frame_size - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate longer sequences
        mfcc = mfcc[:, :frame_size]
    
    return mfcc.T  # Shape: (4687, 40)
Step 2: Model Implementation
pythonimport torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Single Conv1D + MaxPool1D block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class SigmoidMultiHeadAttention(nn.Module):
    """Sigmoid-based MHA for multi-point activation"""
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Sigmoid attention (not softmax!)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.sigmoid(scores)  # Key difference
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Final linear projection
        output = self.W_O(attn_output)
        return output

class HLGNet(nn.Module):
    """Hierarchical Local-Global Network"""
    def __init__(self, input_dim=40):
        super().__init__()
        
        # Local Feature Extraction (3 Conv blocks)
        self.conv1 = ConvBlock(input_dim, 64, kernel_size=3, pool_size=5)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, pool_size=5)
        self.conv3 = ConvBlock(64, 64, kernel_size=3, pool_size=5)
        
        # Global Feature Extraction (Sigmoid-MHA)
        self.mha = SigmoidMultiHeadAttention(d_model=64, num_heads=8)
        
        # Classification
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        # x: (batch, seq_len=4687, input_dim=40)
        
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
        x = self.fc(x)         # → (batch, 1)
        
        return x.squeeze(-1)   # → (batch,)
Step 3: Training Pipeline
pythonimport torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DepressionDataset(Dataset):
    """AVEC 2014 dataset wrapper"""
    def __init__(self, mfcc_list, bdi_scores):
        self.mfcc = mfcc_list      # List of (4687, 40) arrays
        self.scores = bdi_scores   # List of BDI-II scores
    
    def __len__(self):
        return len(self.mfcc)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.mfcc[idx]),
            torch.FloatTensor([self.scores[idx]])
        )

def train_hlgnet(train_loader, val_loader, epochs=100, lr=1e-3):
    """Training loop with MSE loss"""
    model = HLGNet(input_dim=40)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for mfcc, bdi_score in train_loader:
            optimizer.zero_grad()
            
            pred = model(mfcc)
            loss = criterion(pred, bdi_score.squeeze())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for mfcc, bdi_score in val_loader:
                pred = model(mfcc)
                val_mae += torch.abs(pred - bdi_score.squeeze()).sum().item()
        
        val_mae /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.3f}, Val MAE={val_mae:.2f}")
    
    return model
Step 4: Post-Training Quantization
pythonfrom mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_quantization, enable_calibration_woquantization

def quantize_model(model, calib_loader):
    """12-bit quantization with MSE observer"""
    
    # Prepare quantization-aware model
    backend = BackendType.Academic  # Or your target hardware
    model = prepare_by_platform(model, backend)
    
    # Calibration phase
    enable_calibration_woquantization(model)
    model.eval()
    
    with torch.no_grad():
        for mfcc, _ in calib_loader:
            _ = model(mfcc)
    
    # Enable quantization
    enable_quantization(model)
    
    return model
Step 5: TSF (Ternary Step Function) Optimization
For hardware deployment, replace sigmoid with TSF:
pythonclass TSF(nn.Module):
    """Ternary Step Function (hardware-friendly)"""
    def forward(self, x):
        # Assumes x is pre-quantized with scale factor 32
        return torch.where(x < -16, torch.zeros_like(x),
               torch.where(x > 16, torch.ones_like(x),
               0.5 * torch.ones_like(x)))

# Replace sigmoid in attention:
# attn_weights = torch.sigmoid(scores)  # Original
attn_weights = TSF()(scores)            # Hardware-optimized
Step 6: Evaluation
pythondef evaluate(model, test_loader):
    """Compute MAE, RMSE, and 2-class accuracy"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for mfcc, bdi_score in test_loader:
            pred = model(mfcc)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(bdi_score.squeeze().cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Metrics
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    # Two-class accuracy (depressed: BDI-II > 13)
    pred_class = (all_preds > 13).astype(int)
    true_class = (all_targets > 13).astype(int)
    accuracy = (pred_class == true_class).mean()
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    return mae, rmse, accuracy

Training Details
ParameterValueFrame Size4687 framesMFCC Dimensions40Quantization12-bit fixed-pointObserverMSE-basedLoss FunctionMSEOptimizerAdam (lr=1e-3)Batch Size32 (recommended)Epochs100-150