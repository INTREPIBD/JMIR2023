import tensorflow as tf


def cross_entropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=False
        )
    )


def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_mean(
        tf.metrics.sparse_categorical_accuracy(y_true=y_true, y_pred=y_pred)
    )
