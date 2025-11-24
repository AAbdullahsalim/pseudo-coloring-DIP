"""
Professional training script for pseudo-coloring U-Net model.
Features:
- Automatic grayscale conversion from color images
- Validation split
- Model checkpointing
- Training history logging
- Configurable hyperparameters
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from unet_model import build_unet
from losses import get_loss_function
import config
import os
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_generators(batch_size=config.BATCH_SIZE, validation_split=config.VALIDATION_SPLIT):
    """
    Create data generators for training and validation.
    Automatically converts color images to grayscale for perfect pairing.
    
    Args:
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        train_generator, val_generator, num_train_samples, num_val_samples
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1/255.,
        validation_split=validation_split,
        **config.AUGMENTATION
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1/255.,
        validation_split=validation_split
    )
    
    # Training generator
    train_color_gen = train_datagen.flow_from_directory(
        config.COLOR_DIR,
        target_size=config.IMAGE_SIZE,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        subset='training',
        seed=42,
        shuffle=True
    )
    
    # Validation generator
    val_color_gen = val_datagen.flow_from_directory(
        config.COLOR_DIR,
        target_size=config.IMAGE_SIZE,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        subset='validation',
        seed=42,
        shuffle=False
    )
    
    def train_generator():
        """Generator that yields (grayscale_input, color_target) pairs for training."""
        while True:
            color_batch = next(train_color_gen)
            gray_batch = tf.image.rgb_to_grayscale(color_batch)
            yield gray_batch, color_batch
    
    def val_generator():
        """Generator that yields (grayscale_input, color_target) pairs for validation."""
        while True:
            color_batch = next(val_color_gen)
            gray_batch = tf.image.rgb_to_grayscale(color_batch)
            yield gray_batch, color_batch
    
    # Calculate number of samples
    num_train_samples = train_color_gen.samples
    num_val_samples = val_color_gen.samples
    
    return train_generator(), val_generator(), num_train_samples, num_val_samples


def setup_callbacks():
    """Setup training callbacks for model checkpointing and early stopping."""
    callbacks = []
    
    if config.USE_CHECKPOINTS:
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR,
            f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor=config.MONITOR_METRIC,
            save_best_only=config.SAVE_BEST_ONLY,
            mode='min',
            verbose=1,
            save_weights_only=False
        )
        callbacks.append(checkpoint)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor=config.MONITOR_METRIC,
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor=config.MONITOR_METRIC,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # CSV logger
    csv_logger = CSVLogger(
        os.path.join(config.CHECKPOINT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
        append=True
    )
    callbacks.append(csv_logger)
    
    return callbacks


def save_training_info(num_train, num_val):
    """Save training configuration and dataset info."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "model_name": config.MODEL_NAME,
        "image_size": config.IMAGE_SIZE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "steps_per_epoch": config.STEPS_PER_EPOCH,
        "learning_rate": config.LEARNING_RATE,
        "loss_weight_pcc": config.LOSS_WEIGHT_PCC,
        "validation_split": config.VALIDATION_SPLIT,
        "num_train_samples": num_train,
        "num_val_samples": num_val,
        "augmentation": config.AUGMENTATION
    }
    
    info_path = os.path.join(config.CHECKPOINT_DIR, f"training_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining info saved to: {info_path}")
    return info


def main():
    """Main training function."""
    print("=" * 60)
    print("Pseudo-Coloring U-Net Training")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(config.COLOR_DIR):
        print(f"ERROR: Color image directory not found: {config.COLOR_DIR}")
        print("Please ensure your training images are in the correct location.")
        return
    
    # Build model
    print("\n[1/5] Building U-Net model...")
    model = build_unet(input_shape=config.INPUT_SHAPE)
    print(f"Model built with {model.count_params():,} parameters")
    
    # Compile model
    print("\n[2/5] Compiling model...")
    loss_fn = get_loss_function(pcc_weight=config.LOSS_WEIGHT_PCC)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=["mse", "mae"]
    )
    
    # Create data generators
    print("\n[3/5] Creating data generators...")
    train_gen, val_gen, num_train, num_val = make_generators()
    print(f"Training samples: {num_train}")
    print(f"Validation samples: {num_val}")
    
    if num_train == 0:
        print("ERROR: No training images found!")
        return
    
    # Setup callbacks
    print("\n[4/5] Setting up callbacks...")
    callbacks = setup_callbacks()
    
    # Calculate steps
    train_steps = min(config.STEPS_PER_EPOCH, (num_train // config.BATCH_SIZE) + 1)
    val_steps = (num_val // config.BATCH_SIZE) + 1 if num_val > 0 else None
    
    # Save training info
    save_training_info(num_train, num_val)
    
    # Train model
    print("\n[5/5] Starting training...")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Steps per epoch: {train_steps}")
    print(f"Validation steps: {val_steps}")
    print("-" * 60)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=config.EPOCHS,
        validation_data=val_gen if num_val > 0 else None,
        validation_steps=val_steps if num_val > 0 else None,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n" + "=" * 60)
    print("Saving final model...")
    model.save(config.MODEL_NAME)
    print(f"✓ Model saved to: {config.MODEL_NAME}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print("=" * 60)
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

