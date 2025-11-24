"""
Configuration file for pseudo-coloring model.
Modify these settings to adjust training and inference behavior.
"""
import os

# ==================== Data Configuration ====================
DATA_DIR = "data"
COLOR_DIR = os.path.join(DATA_DIR, "color")
GRAY_DIR = os.path.join(DATA_DIR, "gray")

# Image settings
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.2  # 20% of data for validation

# ==================== Model Configuration ====================
MODEL_NAME = "pseudo_color_model.h5"
INPUT_SHAPE = (*IMAGE_SIZE, 1)  # Grayscale input
OUTPUT_CHANNELS = 3  # RGB output

# U-Net architecture
UNET_FILTERS = [32, 64, 128, 256]  # Encoder/decoder filter sizes

# ==================== Training Configuration ====================
EPOCHS = 30  # Increased for better learning
STEPS_PER_EPOCH = 50
LEARNING_RATE = 1e-4
LOSS_WEIGHT_PCC = 0.2  # Reduced to focus more on color accuracy

# Data augmentation
AUGMENTATION = {
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "zoom_range": 0.1
}

# Training callbacks
USE_CHECKPOINTS = True
CHECKPOINT_DIR = "checkpoints"
SAVE_BEST_ONLY = True
MONITOR_METRIC = "val_loss"

# ==================== Inference Configuration ====================
DEFAULT_OUTPUT_DIR = "outputs"
OUTPUT_FORMAT = "png"  # png, jpg, jpeg

# ==================== Paths ====================
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

