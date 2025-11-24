"""
U-Net architecture for pseudo-coloring.
Converts grayscale images to colorized versions.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import config


def conv_block(x, filters, kernel_size=3):
    """
    Convolutional block with two conv layers.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Size of convolution kernel
    
    Returns:
        Output tensor after two convolutions
    """
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    return x


def encoder_block(x, filters):
    """
    Encoder block: conv block + max pooling.
    
    Args:
        x: Input tensor
        filters: Number of filters
    
    Returns:
        skip: Skip connection (before pooling)
        pooled: Pooled output (for next layer)
    """
    skip = conv_block(x, filters)
    pooled = layers.MaxPooling2D((2, 2))(skip)
    return skip, pooled


def decoder_block(x, skip, filters):
    """
    Decoder block: upsampling + concatenation + conv block.
    
    Args:
        x: Input tensor from previous decoder layer
        skip: Skip connection from encoder
        filters: Number of filters
    
    Returns:
        Output tensor
    """
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(input_shape=None, output_channels=None, filters=None):
    """
    Build U-Net model for pseudo-coloring.
    
    Args:
        input_shape: Input shape (H, W, C). Defaults to config.INPUT_SHAPE
        output_channels: Number of output channels. Defaults to config.OUTPUT_CHANNELS
        filters: List of filter sizes for encoder/decoder. Defaults to config.UNET_FILTERS
    
    Returns:
        Compiled Keras model
    """
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if output_channels is None:
        output_channels = config.OUTPUT_CHANNELS
    if filters is None:
        filters = config.UNET_FILTERS
    
    inputs = layers.Input(input_shape)
    
    # Encoder path
    skip1, p1 = encoder_block(inputs, filters[0])
    skip2, p2 = encoder_block(p1, filters[1])
    skip3, p3 = encoder_block(p2, filters[2])
    
    # Bottleneck
    bottleneck = conv_block(p3, filters[3])
    
    # Decoder path
    d1 = decoder_block(bottleneck, skip3, filters[2])
    d2 = decoder_block(d1, skip2, filters[1])
    d3 = decoder_block(d2, skip1, filters[0])
    
    # Output layer
    outputs = layers.Conv2D(
        output_channels,
        1,
        padding="same",
        activation="sigmoid"
    )(d3)
    
    model = Model(inputs, outputs, name="PseudoColorUNet")
    return model


if __name__ == "__main__":
    # Test model building
    print("Building U-Net model...")
    model = build_unet()
    print(f"Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    model.summary()
