from .registry import register

import tensorflow as tf

from timebase.models import utils


@register("bilstm")
def get_model(args, name: str = "BiLSTM"):
    inputs = tf.keras.Input(args.input_shape, name="inputs")

    outputs = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(
            units=args.num_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            dropout=args.dropout,
            recurrent_dropout=args.r_dropout,
            unroll=False,
        ),
        merge_mode="concat",
        name="BiLSTM",
    )(inputs)

    outputs = tf.keras.layers.Dense(units=args.num_classes, name="output")(outputs)
    outputs = utils.Activation(activation="softmax", name="softmax", dtype=tf.float32)(
        outputs
    )

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
