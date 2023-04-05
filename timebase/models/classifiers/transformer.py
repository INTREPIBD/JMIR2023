from .registry import register

import tensorflow as tf

from timebase.models import utils


def transformer_encoder(
    inputs: tf.Tensor,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    activation: str,
    dropout: float = 0.0,
    name: str = "encoder",
):
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"{name}/layer_norm_1"
    )(inputs)
    outputs = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout,
        name=f"{name}/multi_head_attention",
    )(query=outputs, value=outputs)
    outputs = tf.keras.layers.Dropout(dropout, name=f"{name}/dropout_1")(outputs)
    residual = tf.keras.layers.Add(name=f"{name}/add_1")([outputs, inputs])

    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"{name}/layer_norm_2"
    )(residual)
    outputs = tf.keras.layers.Conv1D(
        filters=ff_dim, kernel_size=1, name=f"{name}/conv_1"
    )(outputs)
    outputs = utils.Activation(activation, name=f"{name}/activation_1")(outputs)
    outputs = tf.keras.layers.Dropout(dropout, name=f"{name}/dropout_2")(outputs)
    outputs = tf.keras.layers.Conv1D(
        filters=inputs.shape[-1], kernel_size=1, name=f"{name}/conv_2"
    )(outputs)

    outputs = tf.keras.layers.Add(name=f"{name}/add_2")([outputs, residual])

    return outputs


@register("transformer")
def build_model(args, name: str = "transformer"):
    inputs = tf.keras.Input(args.input_shape, name="inputs")

    outputs = inputs
    for i in range(args.num_encoders):
        outputs = transformer_encoder(
            inputs=outputs,
            head_size=args.head_size,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            activation=args.activation,
            dropout=args.t_dropout,
            name=f"{name}/encoder_{i+1}",
        )

    outputs = tf.keras.layers.GlobalAveragePooling1D(
        data_format="channels_first", name="global_average_pooling"
    )(outputs)

    outputs = tf.keras.layers.Dense(args.num_units, name=f"{name}/dense")(outputs)
    outputs = utils.Activation(args.activation, name=f"{name}/activation")(outputs)
    outputs = tf.keras.layers.Dropout(args.dropout, name=f"{name}/dropout")(outputs)

    outputs = tf.keras.layers.Dense(args.num_classes, name="output")(outputs)
    outputs = utils.Activation("softmax", name="softmax", dtype=tf.float32)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
