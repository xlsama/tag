"""训练配置"""

from pathlib import Path

# 路径配置
TRAIN_DIR = Path(__file__).parent
DATA_FILE = TRAIN_DIR / 'data.jsonl'
LABELS_FILE = TRAIN_DIR / 'labels.json'
OUTPUT_DIR = TRAIN_DIR / 'output'

# 模型配置
MODEL_NAME = 'unsloth/Qwen3-4B-bnb-4bit'
MAX_SEQ_LENGTH = 512

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]

# 训练配置
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
SAVE_STEPS = 50
LOGGING_STEPS = 10
