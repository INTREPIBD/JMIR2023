import tensorflow as tf


class Optimizer:
    """Adam optimizer wrapper with mixed precision scaling"""

    def __init__(self, args, model: tf.keras.Model):
        self.mixed_precision = args.mixed_precision
        self.model = model

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        if self.mixed_precision:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

    def get_weights(self):
        return self.optimizer.get_weights()

    def set_weights(self, weights):
        self.optimizer.apply_gradients(
            zip(
                [tf.zeros_like(v) for v in self.model.trainable_variables],
                self.model.trainable_variables,
            )
        )
        self.optimizer.set_weights(weights)

    def get_scaled_loss(self, loss: tf.Tensor):
        """Get scaled loss if mixed precision is enabled."""
        if self.mixed_precision:
            loss = self.optimizer.get_scaled_loss(loss)
        return loss

    def get_unscaled_gradients(self, scaled_gradients):
        return self.optimizer.get_unscaled_gradients(scaled_gradients)

    def minimize(self, loss: tf.Tensor, tape: tf.GradientTape):
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.mixed_precision:
            gradients = self.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
