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


def full_loss(y_true, y_pred, pcc_weight=0.3):
    """
    Combined loss function: MSE + weighted PCC loss.
    
    Args:
        y_true: Ground truth color images [batch, H, W, 3]
        y_pred: Predicted color images [batch, H, W, 3]
        pcc_weight: Weight for PCC loss component (default: 0.3)
    
    Returns:
        Combined loss value
    """
    # Mean Squared Error
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Perceptual Color Consistency loss
    pcc = pcc_loss(y_true, y_pred)
    
    # Combined loss
    return mse + pcc_weight * pcc


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

