# Pseudo-Coloring with Deep Learning

A professional implementation of pseudo-coloring (grayscale to color image conversion) using a U-Net deep learning model. This project uses a custom Perceptual Color Consistency (PCC) loss function to produce realistic colorized images.

## Features

- **U-Net Architecture**: Deep encoder-decoder network for image-to-image translation
- **Custom Loss Function**: Combined MSE + PCC loss for better color consistency
- **Automatic Data Pairing**: Converts color images to grayscale during training for perfect pairing
- **Validation & Checkpointing**: Saves best models and training history
- **Batch Processing**: Process multiple images at once
- **Command-Line Interface**: Easy-to-use inference script
- **Configurable**: All hyperparameters in `config.py`

## Project Structure

```
.
├── config.py           # Configuration and hyperparameters
├── losses.py           # Custom loss functions
├── unet_model.py       # U-Net architecture
├── train.py            # Training script
├── inference.py        # Inference script
├── data/
│   └── color/
│       └── class1/     # Put your color training images here
└── outputs/            # Generated colorized images
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare your data:**
   - Place all your color training images in `data/color/class1/`
   - The training script will automatically convert them to grayscale

## Usage

### Training

Train the model on your dataset:

```bash
python train.py
```

**Configuration:**
- Modify `config.py` to adjust hyperparameters
- Training images should be in `data/color/class1/`
- Model will be saved as `pseudo_color_model.h5`
- Checkpoints saved in `checkpoints/` directory

**Key Settings in `config.py`:**
- `EPOCHS`: Number of training epochs (default: 20)
- `BATCH_SIZE`: Batch size (default: 4)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `VALIDATION_SPLIT`: Fraction for validation (default: 0.2)
- `LOSS_WEIGHT_PCC`: Weight for PCC loss component (default: 0.3)

### Inference

**Single Image:**
```bash
python inference.py path/to/grayscale/image.jpg
```

**With Custom Output:**
```bash
python inference.py path/to/image.jpg -o output_path.png
```

**Batch Processing:**
```bash
python inference.py path/to/directory/ --batch
```

**Using Different Model:**
```bash
python inference.py image.jpg -m path/to/model.h5
```

**Default (if no arguments):**
```bash
python inference.py
# Uses: data/gray/class1/grey image.jpeg
```

## How It Works

1. **Training:**
   - Loads color images from `data/color/class1/`
   - Automatically converts to grayscale during training
   - Ensures perfect pairing: grayscale input → color target (same image)
   - Uses data augmentation for better generalization
   - Saves best model based on validation loss

2. **Inference:**
   - Loads trained model
   - Preprocesses grayscale input image
   - Generates colorized output
   - Handles RGB/BGR conversion automatically

## Model Architecture

- **Input**: Grayscale image (256×256×1)
- **Output**: Colorized image (256×256×3)
- **Architecture**: U-Net with skip connections
- **Encoder**: 3 blocks with filters [32, 64, 128]
- **Bottleneck**: 256 filters
- **Decoder**: 3 blocks with skip connections

## Loss Function

Combined loss: `MSE + 0.3 × PCC Loss`

- **MSE**: Mean Squared Error for pixel-level accuracy
- **PCC Loss**: Perceptual Color Consistency - penalizes color gradient differences

## Tips for Better Results

1. **More Training Data**: 50+ diverse images recommended
2. **Diverse Images**: Include different subjects, colors, and lighting
3. **More Epochs**: Increase `EPOCHS` in `config.py` for better convergence
4. **Adjust Learning Rate**: Lower for fine-tuning, higher for faster training
5. **Monitor Validation**: Check `checkpoints/training_log_*.csv` for training progress

## Output

- **Training**: Model saved as `pseudo_color_model.h5`
- **Checkpoints**: Best models in `checkpoints/` directory
- **Logs**: Training history in CSV format
- **Inference**: Colorized images in `outputs/` directory (or specified path)

## Troubleshooting

**Model not found:**
- Train the model first: `python train.py`

**No images found:**
- Ensure images are in `data/color/class1/`
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

**Poor colorization:**
- Add more diverse training images
- Increase training epochs
- Adjust `LOSS_WEIGHT_PCC` in `config.py`

**Memory errors:**
- Reduce `BATCH_SIZE` in `config.py`
- Use smaller images (modify `IMAGE_SIZE`)

## License

This project is for educational purposes.

## Author

Deep Learning Pseudo-Coloring Project

