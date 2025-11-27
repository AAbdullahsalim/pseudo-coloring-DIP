"""
Custom loss functions for pseudo-coloring model.
"""
import tensorflow as tf


def pcc_loss(y_true, y_pred):
    """
    Perceptual Color Consistency (PCC) Loss.
    Penalizes differences in color gradients between neighboring pixels.
    
    Args:
        y_true: Ground truth color images [batch, H, W, 3]
        y_pred: Predicted color images [batch, H, W, 3]
    
    Returns:
        Scalar loss value
    """
    # Compute vertical gradients (differences between neighbor pixels)
    dy_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    dy_pred = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    
    return tf.reduce_mean(tf.abs(dy_true - dy_pred))


def local_color_diversity_loss(y_true, y_pred):
    """
    Penalizes uniform color application across the image.
    Encourages local color diversity - different regions should have different colors.
    Uses spatial variance to detect uniform color application.
    
    Args:
        y_true: Ground truth color images [batch, H, W, 3]
        y_pred: Predicted color images [batch, H, W, 3]
    
    Returns:
        Loss value encouraging local color diversity
    """
    # Calculate spatial variance of colors (how much colors vary across the image)
    # If the entire image has the same color, variance will be low
    
    # For each color channel, calculate spatial variance
    true_spatial_var = []
    pred_spatial_var = []
    
    for c in range(3):  # For each RGB channel
        # Calculate variance across spatial dimensions (H, W)
        true_channel = y_true[:, :, :, c]
        pred_channel = y_pred[:, :, :, c]
        
        # Spatial variance (how much the color varies across the image)
        true_var = tf.reduce_mean(tf.math.reduce_std(true_channel, axis=[1, 2]))
        pred_var = tf.reduce_mean(tf.math.reduce_std(pred_channel, axis=[1, 2]))
        
        true_spatial_var.append(true_var)
        pred_spatial_var.append(pred_var)
    
    # Average variance across channels
    true_avg_var = tf.reduce_mean(true_spatial_var)
    pred_avg_var = tf.reduce_mean(pred_spatial_var)
    
    # Penalize if predicted has low variance (uniform colors across image)
    # Encourage predicted variance to match true variance
    variance_diff = tf.abs(true_avg_var - pred_avg_var)
    uniform_penalty = tf.maximum(0.0, 0.02 - pred_avg_var) * 15.0  # Stronger penalty if too uniform
    
    diversity_loss = variance_diff + uniform_penalty
    
    return diversity_loss


def full_loss(y_true, y_pred, pcc_weight=0.15):
    """
    Combined loss function: MSE + weighted PCC loss + local color diversity.
    Enhanced to preserve color vibrancy, prevent gray outputs, and encourage local color variation.
    
    Args:
        y_true: Ground truth color images [batch, H, W, 3]
        y_pred: Predicted color images [batch, H, W, 3]
        pcc_weight: Weight for PCC loss component (default: 0.15)
    
    Returns:
        Combined loss value
    """
    # Mean Squared Error - primary loss for color accuracy
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Perceptual Color Consistency loss - for smooth color transitions
    pcc = pcc_loss(y_true, y_pred)
    
    # Strong penalty for gray/desaturated predictions
    y_true_gray = tf.reduce_mean(y_true, axis=-1, keepdims=True)
    y_pred_gray = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    
    # Color deviation from grayscale (saturation measure)
    true_color_dev = tf.reduce_mean(tf.abs(y_true - y_true_gray))
    pred_color_dev = tf.reduce_mean(tf.abs(y_pred - y_pred_gray))
    
    # Strong penalty if prediction is too gray (low color deviation)
    gray_penalty = tf.maximum(0.0, 0.05 - pred_color_dev) * 10.0
    
    # Encourage similar color saturation
    sat_loss = tf.abs(true_color_dev - pred_color_dev)
    
    # NEW: Local color diversity loss - prevents uniform color application
    diversity_loss = local_color_diversity_loss(y_true, y_pred)
    
    # ADDITIONAL: Direct penalty for uniform color application
    # Calculate if entire image has similar colors (stronger penalty)
    # Use fixed indices for 256x256 images
    pred_center = y_pred[:, 64:192, 64:192, :]  # Center region [batch, 128, 128, 3]
    
    # Edge regions - calculate mean for each separately
    top_edge = y_pred[:, :64, :, :]      # Top [batch, 64, 256, 3]
    bottom_edge = y_pred[:, 192:, :, :]  # Bottom [batch, 64, 256, 3]
    left_edge = y_pred[:, :, :64, :]     # Left [batch, 256, 64, 3]
    right_edge = y_pred[:, :, 192:, :]  # Right [batch, 256, 64, 3]
    
    # Calculate mean color for each region (reduce over spatial dimensions)
    center_mean = tf.reduce_mean(pred_center, axis=[1, 2])  # [batch, 3]
    top_mean = tf.reduce_mean(top_edge, axis=[1, 2])        # [batch, 3]
    bottom_mean = tf.reduce_mean(bottom_edge, axis=[1, 2])  # [batch, 3]
    left_mean = tf.reduce_mean(left_edge, axis=[1, 2])       # [batch, 3]
    right_mean = tf.reduce_mean(right_edge, axis=[1, 2])    # [batch, 3]
    
    # Average of all edge means
    edges_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4.0  # [batch, 3]
    
    # Penalize if center and edges are too similar (uniform color)
    # Calculate difference per batch, then average
    color_diff = tf.abs(center_mean - edges_mean)  # [batch, 3]
    color_similarity = tf.reduce_mean(color_diff)  # Scalar
    uniform_penalty_strong = tf.maximum(0.0, 0.1 - color_similarity) * 20.0  # Very strong penalty
    
    # Combined loss - MSE is primary, with STRONG penalties for gray/uniform outputs
    return mse + pcc_weight * pcc + 0.2 * sat_loss + gray_penalty + 0.5 * diversity_loss + uniform_penalty_strong


def get_loss_function(pcc_weight=0.3):
    """
    Returns a loss function with specified PCC weight.
    
    Args:
        pcc_weight: Weight for PCC loss component
    
    Returns:
        Loss function compatible with Keras
    """
    def loss_fn(y_true, y_pred):
        return full_loss(y_true, y_pred, pcc_weight=pcc_weight)
    
    return loss_fn

