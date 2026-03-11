"""
Focal loss for imbalanced binary classification.

Focal Loss down-weights well-classified (easy) examples and focuses
training on hard examples.  This is particularly useful for the
EATD-Corpus dataset where the depression class is a minority (~18%).

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import tensorflow as tf


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Create a focal loss function for binary classification.

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Balancing factor for the positive class (0–1).

    Returns:
        A Keras-compatible loss function.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        focal = alpha * tf.pow(1.0 - pt, gamma) * bce
        return focal

    loss_fn.__name__ = "focal_loss"
    return loss_fn
