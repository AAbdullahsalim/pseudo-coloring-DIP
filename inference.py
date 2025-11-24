"""
Professional inference script for pseudo-coloring model.
Features:
- Command-line interface
- Batch processing
- Multiple output formats
- Better error handling
- Progress tracking
"""
import cv2
import numpy as np
import tensorflow as tf
import os
import argparse
from pathlib import Path
from losses import pcc_loss, full_loss
import config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PseudoColorizer:
    """Class for colorizing grayscale images using trained U-Net model."""
    
    def __init__(self, model_path=None):
        """
        Initialize the colorizer with a trained model.
        
        Args:
            model_path: Path to the trained model file. If None, uses config.MODEL_NAME
        """
        self.model_path = model_path or config.MODEL_NAME
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model with custom loss functions."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please train the model first using: python train.py"
            )
        
        print(f"Loading model from: {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "full_loss": full_loss,
                    "pcc_loss": pcc_loss
                },
                compile=False
            )
            print("✓ Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, img_path):
        """
        Preprocess a grayscale image for model input.
        
        Args:
            img_path: Path to grayscale image
        
        Returns:
            Preprocessed image array [1, H, W, 1]
        """
        # Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # Resize and normalize
        img_resized = cv2.resize(img, config.IMAGE_SIZE) / 255.0
        
        # Add batch and channel dimensions
        img_batch = img_resized.reshape(1, *config.IMAGE_SIZE, 1)
        
        return img_batch
    
    def colorize(self, img_path, output_path=None):
        """
        Colorize a single grayscale image.
        
        Args:
            img_path: Path to input grayscale image
            output_path: Path to save output. If None, auto-generates name
        
        Returns:
            Colorized image array [H, W, 3] in RGB format
        """
        # Preprocess
        img_batch = self.preprocess_image(img_path)
        
        # Predict
        pred = self.model.predict(img_batch, verbose=0)[0]
        
        # Convert to uint8
        pred_img = (np.clip(pred, 0, 1) * 255).astype("uint8")
        
        # Generate output path if not provided
        if output_path is None:
            input_name = Path(img_path).stem
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"{input_name}_colorized.{config.OUTPUT_FORMAT}"
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(output_path, pred_img_bgr)
        
        return pred_img, output_path
    
    def colorize_batch(self, img_paths, output_dir=None):
        """
        Colorize multiple images in batch.
        
        Args:
            img_paths: List of paths to grayscale images
            output_dir: Directory to save outputs. If None, uses config.OUTPUT_DIR
        
        Returns:
            List of (output_image, output_path) tuples
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        total = len(img_paths)
        
        print(f"Processing {total} images...")
        for i, img_path in enumerate(img_paths, 1):
            try:
                input_name = Path(img_path).stem
                output_path = os.path.join(
                    output_dir,
                    f"{input_name}_colorized.{config.OUTPUT_FORMAT}"
                )
                
                pred_img, saved_path = self.colorize(img_path, output_path)
                results.append((pred_img, saved_path))
                
                print(f"[{i}/{total}] ✓ {Path(img_path).name} -> {Path(saved_path).name}")
            except Exception as e:
                print(f"[{i}/{total}] ✗ Error processing {img_path}: {str(e)}")
                results.append((None, None))
        
        return results


def find_images(directory, extensions=None):
    """Find all image files in a directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).rglob(f"*{ext}"))
        image_paths.extend(Path(directory).rglob(f"*{ext.upper()}"))
    
    return [str(p) for p in image_paths]


def main():
    """Main inference function with CLI."""
    parser = argparse.ArgumentParser(
        description="Colorize grayscale images using trained U-Net model"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input grayscale image path or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path or directory (default: auto-generated)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help=f"Path to model file (default: {config.MODEL_NAME})"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in input directory"
    )
    
    args = parser.parse_args()
    
    # Initialize colorizer
    try:
        colorizer = PseudoColorizer(model_path=args.model)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    # Process input
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Batch processing
        if not input_path.is_dir():
            print(f"ERROR: {args.input} is not a directory")
            return 1
        
        img_paths = find_images(input_path)
        if not img_paths:
            print(f"No images found in {input_path}")
            return 1
        
        output_dir = args.output or config.OUTPUT_DIR
        colorizer.colorize_batch(img_paths, output_dir)
        
    else:
        # Single image
        if not input_path.exists():
            print(f"ERROR: Image not found: {args.input}")
            return 1
        
        try:
            pred_img, output_path = colorizer.colorize(args.input, args.output)
            print(f"\n✓ Colorization complete!")
            print(f"  Input:  {args.input}")
            print(f"  Output: {output_path}")
        except Exception as e:
            print(f"ERROR: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    # If run without arguments, use default behavior
    import sys
    if len(sys.argv) == 1:
        # Default: process the example image
        default_image = "data/gray/class1/grey image.jpeg"
        if os.path.exists(default_image):
            print("No arguments provided. Using default image...")
            sys.argv = ["inference.py", default_image]
        else:
            print("Usage: python inference.py <image_path> [-o output_path]")
            print("       python inference.py <directory> --batch")
            sys.exit(1)
    
    exit(main())
