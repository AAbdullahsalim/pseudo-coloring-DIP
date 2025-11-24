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


def full_loss(y_true, y_pred, pcc_weight=0.2):
    """
    Combined loss function: MSE + weighted PCC loss.
    Enhanced to preserve color vibrancy.
    
    Args:
        y_true: Ground truth color images [batch, H, W, 3]
        y_pred: Predicted color images [batch, H, W, 3]
        pcc_weight: Weight for PCC loss component (default: 0.2)
    
    Returns:
        Combined loss value
    """
    # Mean Squared Error - primary loss for color accuracy
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Perceptual Color Consistency loss - for smooth color transitions
    pcc = pcc_loss(y_true, y_pred)
    
    # Additional: Penalize overly desaturated predictions
    # Calculate saturation for true and predicted
    y_true_gray = tf.reduce_mean(y_true, axis=-1, keepdims=True)
    y_pred_gray = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    
    # Saturation = deviation from grayscale
    true_sat = tf.reduce_mean(tf.abs(y_true - y_true_gray))
    pred_sat = tf.reduce_mean(tf.abs(y_pred - y_pred_gray))
    
    # Encourage similar saturation levels (prevents overly dull outputs)
    sat_loss = tf.abs(true_sat - pred_sat)
    
    # Combined loss
    return mse + pcc_weight * pcc + 0.1 * sat_loss


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

