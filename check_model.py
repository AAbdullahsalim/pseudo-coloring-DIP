"""
Quick diagnostic script to check model and data setup.
"""
import os
import tensorflow as tf
from pathlib import Path
import config

print("=" * 60)
print("Model & Data Diagnostic")
print("=" * 60)

# Check model
print("\n[1] Model Check:")
model_path = config.MODEL_NAME
if os.path.exists(model_path):
    print(f"  ✓ Model exists: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"  ✓ Model loaded successfully")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Output shape: {model.output_shape}")
        print(f"  - Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
else:
    print(f"  ✗ Model not found: {model_path}")
    print("  → Run: python train.py")

# Check training data
print("\n[2] Training Data Check:")
color_dir = config.COLOR_DIR
if os.path.exists(color_dir):
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(color_dir).rglob(f"*{ext}"))
        image_files.extend(Path(color_dir).rglob(f"*{ext.upper()}"))
    
    num_images = len(image_files)
    print(f"  ✓ Found {num_images} color images in {color_dir}")
    
    if num_images == 0:
        print("  ✗ No images found! Add images to train the model.")
    elif num_images < 10:
        print(f"  ⚠ Warning: Only {num_images} images. More images (20+) recommended for better results.")
    else:
        print(f"  ✓ Good! {num_images} images should be sufficient.")
else:
    print(f"  ✗ Directory not found: {color_dir}")

# Check if model needs retraining
print("\n[3] Recommendation:")
if os.path.exists(model_path):
    print("  → If output looks wrong, RETRAIN with: python train.py")
    print("  → The old model might have been trained with incorrect data pairing")
else:
    print("  → Train the model: python train.py")

print("\n" + "=" * 60)

