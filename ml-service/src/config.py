"""Central project configuration — single source of truth for all constants.

Import from this module rather than defining constants in multiple files.
"""

# ── Audio ─────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000
DURATION: float = 5.0
AUDIO_LENGTH: int = int(SAMPLE_RATE * DURATION)  # 80_000

# ── STFT / mel filterbank ─────────────────────────────────────────────
N_FFT: int = 512
HOP_LENGTH: int = 256
N_MELS: int = 128
N_MFCC: int = 13
F_MIN: float = 0.0
F_MAX: float = SAMPLE_RATE / 2  # 8000.0 Hz
EXPECTED_TIME_STEPS: int = 313  # floor((AUDIO_LENGTH - N_FFT) / HOP_LENGTH) + 1

# ── Training defaults ─────────────────────────────────────────────────
BATCH_SIZE: int = 32
EPOCHS: int = 100
LEARNING_RATE: float = 0.001
DROPOUT_RATE: float = 0.3
EARLY_STOP_PATIENCE: int = 10
REDUCE_LR_PATIENCE: int = 5
TEST_SIZE: float = 0.15
VAL_SIZE: float = 0.15
RANDOM_STATE: int = 42

# ── Paths (relative to project root) ─────────────────────────────────
DATA_DIR: str = "data/raw/voice_data"
DEPRESSION_DIR: str = "depression1"
NORMAL_DIR: str = "normal1"
MODEL_DIR: str = "artifacts/models"
