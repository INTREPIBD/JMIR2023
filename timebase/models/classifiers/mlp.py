from .registry import register

import tensorflow as tf


from timebase.models import utils


@register("mlp")
def get_model(args, name: str = "mlp"):
    inputs = tf.keras.Input(args.input_shape, name="inputs")

    outputs = tf.keras.layers.Flatten(name="flatten")(inputs)

    outputs = tf.keras.layers.Dense(units=args.num_units, name="dense1")(outputs)
    outputs = tf.keras.layers.Dropout(rate=args.dropout, name="dropout1")(outputs)
    outputs = utils.Activation(activation=args.activation, name="activation1")(outputs)

    outputs = tf.keras.layers.Dense(units=args.num_units // 2, name="dense2")(outputs)
    outputs = tf.keras.layers.Dropout(rate=args.dropout, name="dropout2")(outputs)
    outputs = utils.Activation(activation=args.activation, name="activation2")(outputs)

    outputs = tf.keras.layers.Dense(units=args.num_units // 3, name="dense3")(outputs)
    outputs = tf.keras.layers.Dropout(rate=args.dropout, name="dropout3")(outputs)
    outputs = utils.Activation(activation=args.activation, name="activation3")(outputs)

    outputs = tf.keras.layers.Dense(units=args.num_classes, name="output")(outputs)
    outputs = utils.Activation(activation="softmax", name="softmax", dtype=tf.float32)(
        outputs
    )

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
