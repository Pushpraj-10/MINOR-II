"""Central project configuration — single source of truth for all constants.

Import from this module rather than defining constants in multiple files.
"""

# ── Audio ─────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000       # 16,000 samples per second — standard for speech
DURATION: float = 5.0          # we cut every audio clip to exactly 5 seconds
AUDIO_LENGTH: int = int(SAMPLE_RATE * DURATION)  # 80,000 raw samples per clip

# ── STFT / mel filterbank ─────────────────────────────────────────────
# STFT (Short-Time Fourier Transform) breaks audio into small windows and
# measures the frequency content of each window.  Think of it like scanning
# a spectrogram column by column.
N_FFT: int = 512               # each window is 512 samples wide (~32 ms)
HOP_LENGTH: int = 256          # windows overlap — slide 256 samples at a time
N_MELS: int = 128              # number of mel frequency bins (for mel spectrogram)
N_MFCC: int = 13               # how many MFCC coefficients to keep per time frame
F_MIN: float = 0.0             # lowest frequency to analyse (Hz)
F_MAX: float = SAMPLE_RATE / 2  # highest frequency = Nyquist limit = 8000 Hz
EXPECTED_TIME_STEPS: int = 313  # number of time columns: floor((80000-512)/256)+1

# ── Training defaults ─────────────────────────────────────────────────
BATCH_SIZE: int = 32           # process 32 audio clips at once during training
EPOCHS: int = 100              # maximum training rounds (early stopping may stop sooner)
LEARNING_RATE: float = 0.001   # how big each weight update step is
DROPOUT_RATE: float = 0.3      # fraction of neurons randomly switched off (prevents overfitting)
EARLY_STOP_PATIENCE: int = 10  # stop if validation AUC doesn't improve for 10 epochs
REDUCE_LR_PATIENCE: int = 5    # halve learning rate if no improvement for 5 epochs
TEST_SIZE: float = 0.15        # 15% of data goes to final test
VAL_SIZE: float = 0.15         # 15% of data goes to validation during training
RANDOM_STATE: int = 42         # seed for reproducibility — same splits every run

# ── Paths (relative to project root) ─────────────────────────────────
DATA_DIR: str = "data/raw/voice_data"        # raw DATASET_1 audio files
DEPRESSION_DIR: str = "depression1"          # subfolder with depressed recordings
NORMAL_DIR: str = "normal1"                  # subfolder with normal recordings
MODEL_DIR: str = "artifacts/models"          # where trained models are saved

# EATD-Corpus
EATD_CORPUS_DIR: str = "data/raw/EATD-Corpus/EATD-Corpus"  # clinical interview dataset

# ── HLG-Net configuration ────────────────────────────────────────────
# HLG-Net uses 40-D MFCC at 100 Hz frame rate (10ms hop) as input.
# Conv1D blocks reduce 4687 frames → 37 features, then sigmoid MHA
# captures global patterns for depression severity regression.
HLGNET_N_MFCC: int = 40           # MFCC coefficients (richer than 13)
HLGNET_FRAME_SIZE: int = 4687     # fixed input length in frames
HLGNET_HOP_LENGTH: int = 160      # 10ms hop at 16kHz → 100 Hz frame rate
HLGNET_N_FFT: int = 400           # 25ms window at 16kHz
HLGNET_D_MODEL: int = 64          # hidden dimension for conv + attention
HLGNET_NUM_HEADS: int = 8         # attention heads in sigmoid MHA
HLGNET_EPOCHS: int = 100          # training epochs
HLGNET_LR: float = 1e-3           # Adam learning rate
HLGNET_BATCH_SIZE: int = 32       # batch size

# ── Dataset paths ─────────────────────────────────────────────────────
DAICWOZ_DIR: str = "data/raw/DAICWOZ"                     # AVEC 2017 depression corpus
DEPRESSION_DATASET_DIR: str = "data/raw/dataset-depression"  # RAVDESS-based acted emotion
RAW_RAVDESS_DIR: str = "data/raw/The Ryerson Audio-Visual Dataset" # Original RAVDESS
# EATD_CORPUS_DIR already defined above as EATD_CORPUS_DIR: str = "data/raw/EATD-Corpus/EATD-Corpus"
