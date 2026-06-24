"""
Model architectures for survival analysis.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .loss import weibull_loss


def build_bert_pca(
    n_features: int,
    seq_length: int,
    learning_rate: float,
    num_encoder_layers: int,
    intermediate_dim: int,
    num_heads: int,
    num_conv_blocks: int,
    filters: int,
    kernel_size: int,
    pool_strides: int,
    pool_size: int,
    num_dense_layers: int,
    dense_units: int,
    activation: str = 'relu',
    norm_epsilon: float = 1e-5,
    dropout: float = 0.2,
    gamma: float = 0.001
) -> keras.Model:
    """
    Build the BERT-PCA model for survival analysis.
    
    The model combines:
    - Transformer encoder layers for sequence modeling
    - 1D convolutional blocks for feature extraction
    - Dense layers for final prediction
    - Outputs Weibull distribution parameters (alpha, beta)
    
    Parameters
    ----------
    n_features : int
        Number of input features
    seq_length : int
        Sequence length
    learning_rate : float
        Learning rate for optimizer
    num_encoder_layers : int
        Number of transformer encoder layers
    intermediate_dim : int
        Dimension of feedforward network in transformer
    num_heads : int
        Number of attention heads
    num_conv_blocks : int
        Number of 1D convolutional blocks
    filters : int
        Number of filters in convolutional layers
    kernel_size : int
        Kernel size for convolutional layers
    pool_strides : int
        Stride for pooling layers
    pool_size : int
        Pool size for pooling layers
    num_dense_layers : int
        Number of dense layers
    dense_units : int
        Number of units in dense layers
    activation : str
        Activation function for dense layers (default: 'relu')
    norm_epsilon : float
        Epsilon for layer normalization (default: 1e-5)
    dropout : float
        Dropout rate (default: 0.2)
    gamma : float
        Loss weight parameter (default: 0.001)
    
    Returns
    -------
    keras.Model
        Compiled Keras model
    """
    # Input layer
    input_layer = keras.Input(
        shape=(n_features, seq_length,),
        dtype=tf.float64,
        name="input"
    )
    x = input_layer

    # Transpose for Transformer: [batch, seq_len, features]
    input_T = keras.layers.Permute((2, 1), name="transpose_input")(x)

    # --- Transformer Encoder Layers ---
    for i in range(num_encoder_layers):
        # Multi-Head Attention
        attn_output, attn_scores = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=x.shape[-1],
            dropout=dropout,
            name=f"mha_{i}"
        )(x, x, return_attention_scores=True)

        # Add & Norm
        x = layers.Add(name=f"mha_add_{i}")([x, attn_output])
        x = layers.LayerNormalization(
            epsilon=norm_epsilon,
            name=f"mha_norm_{i}"
        )(x)

        # Feedforward Network
        ffn_output = layers.Dense(
            intermediate_dim,
            activation='relu',
            name=f"ffn_{i}_dense1"
        )(x)
        ffn_output = layers.Dense(
            x.shape[-1],
            name=f"ffn_{i}_dense2"
        )(ffn_output)
        ffn_output = layers.Dropout(
            dropout,
            name=f"ffn_{i}_dropout"
        )(ffn_output)

        # Add & Norm
        x = layers.Add(name=f"ffn_add_{i}")([x, ffn_output])
        x = layers.LayerNormalization(
            epsilon=norm_epsilon,
            name=f"ffn_norm_{i}"
        )(x)

    # Transpose back for Conv1D layers
    x = keras.layers.Permute((2, 1), name="transpose_back")(x)

    # Residual connection with transposed input
    x = keras.layers.Add(name="residual_add_input")([x, input_T])

    # --- 1D Convolutional Blocks ---
    for i in range(num_conv_blocks):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name=f"conv1d_{i}"
        )(x)
        x = layers.AveragePooling1D(
            pool_size=pool_size,
            strides=pool_strides,
            name=f"avg_pool_{i}"
        )(x)

    # --- Dense Layers ---
    x = keras.layers.Flatten(name="flatten")(x)

    for i in range(num_dense_layers):
        x = layers.Dense(
            dense_units,
            activation=activation,
            name=f"dense_{i}"
        )(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    # --- Output Weibull Parameters ---
    alpha = keras.layers.Dense(
        1,
        activation='elu',
        name="alpha",
        use_bias=False
    )(x)
    beta = keras.layers.Dense(
        1,
        activation='elu',
        name="beta",
        use_bias=False
    )(x)
    output = keras.layers.Concatenate(name="weibull_params")([alpha, beta])

    # --- Build and Compile Model ---
    model = keras.Model(inputs=input_layer, outputs=output, name="BERT_PCA")
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=weibull_loss
    )

    return model
