from .registry import register

import tensorflow as tf

from timebase.models import utils


@register("gru")
def get_model(args, name: str = "gru"):
    inputs = tf.keras.Input(args.input_shape, name="inputs")

    outputs = tf.keras.layers.GRU(
        units=args.num_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=args.dropout,
        recurrent_dropout=args.r_dropout,
        return_sequences=False,
        name="gru",
    )(inputs)

    outputs = tf.keras.layers.Dense(units=args.num_classes, name="output")(outputs)
    outputs = utils.Activation(activation="softmax", name="softmax", dtype=tf.float32)(
        outputs
    )

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
