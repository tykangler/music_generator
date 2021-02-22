# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Model Architecture
# 
# 1276 total files, 962 train files (75%)

# %%
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Multi-Head Relative Attention

# %%
class MultiHeadRelativeAttention(keras.layers.Layer):
    """
    An implementation of the multi head attention mechanism, with relative position representations. 
    This impl requires that the Q, K, and V all have the same shape of (..., seqlen, dims)
    """
    def __init__(self, num_heads, dims, max_relative_pos, max_seqlen, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dims = dims

        assert self.dims % self.num_heads == 0,             """qkv tensors must be able to be represented as 
            a list of tensors with length num_heads (dims % num_heads == 0)"""
        self.head_dims = self.dims // self.num_heads
        
        self.num_rprs = max_relative_pos * 2 + 1
        self.rpr_embedding = keras.layers.Embedding(self.num_rprs, self.head_dims)
        self.rpr_lookup = self._generate_rpr_lookup(max_seqlen, max_relative_pos)

        # split to num_heads at call
        self.weights_q = keras.layers.Dense(dims)
        self.weights_k = keras.layers.Dense(dims)
        self.weights_v = keras.layers.Dense(dims)

        # concatenated heads passed as input
        self.concat_head_weights = keras.layers.Dense(dims)

    def _generate_rpr_lookup(self, input_length, max_relative_pos):
        x = np.arange(input_length)
        x = tf.expand_dims(x, axis=0) - tf.expand_dims(x, axis=1)
        x = tf.clip_by_value(x, -max_relative_pos, max_relative_pos)
        return x + max_relative_pos

    def _project_to_heads(self, x):
        """
        projects x to `num_heads` number of heads
        params:
            x: tensor of shape (batch, seqlen, dims)
        returns:
            tensor of shape (batch, num_heads, seqlen, head_dims). Each sequence in the batch is 
                split column wise so that each head attends to a subsection of dims in each sequence.
        
        example:
            x with shape (batch=2, seqlen=3, dims=4)
            [[[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]],

             [[12, 13, 14, 15],
              [16, 17, 18, 19],
              [20, 21, 22, 23]]]
            split to (batch=2, seqlen=3, num_heads=2, head_dims=2)
            [[[[ 0,  1],
               [ 2,  3]],
      
              [[ 4,  5],
               [ 6,  7]],
      
              [[ 8,  9],
               [10, 11]]],
       
             [[[12, 13],
               [14, 15]],
      
              [[16, 17],
               [18, 19]],
      
              [[20, 21],
               [22, 23]]]])
            transpose to (batch=2, num_heads=2, seqlen=3, head_dims=2)
            [[[[ 0,  1],
               [ 4,  5],
               [ 8,  9]],
         
              [[ 2,  3],
               [ 6,  7],
               [10, 11]]],
         
             [[[12, 13],
               [16, 17],
               [20, 21]],
         
              [[14, 15],
               [18, 19],
               [22, 23]]]])
            
        """
        x = tf.reshape(x, (x.shape[0], -1, self.num_heads, self.head_dims))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _compute_relative(self, x, embeddings, transpose_embeddings=False):
        """
        computes relational attention across all input sequences in heads and batches. 
        
        For query, `x` is transposed to (seqlen, batch, heads, head_dims), reshaped to (seqlen, batch * heads, head_dims),
        then multiplied with embeddings with shape (seqlen, head_dims, seqlen) (after transpose). The resulting
        shape is (seqlen, batch * heads, seqlen). 
        
        For attn_weights, `x` is transposed to (seqlen, batch, heads, seqlen), reshaped to (seqlen, batch * heads, seqlen), 
        then multiplied with embeddings with shape (seqlen, seqlen, head_dims). The resulting shape is 
        (seqlen, batch * heads, head_dims). In both cases, the result wil be reshaped back to (seqlen, batch, heads, ...),
        then transposed back to (batch, heads, seqlen, ...). This approach avoids broadcasting.
        :params:
            x: tensor (query or attn weights) with shape (batch, heads, seqlen, ...)
            embeddings: learned rpr embeddings with shape (seqlen, seqlen, head_dims)
            transpose_embeddings: Whether to transpose the embeddings argument, select true for query 
        """
        x = tf.transpose(x, perm=[2, 0, 1, 3]) # (seqlen, batch, heads, head_dims)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1])) # (seqlen, batch * heads, head_dims)
        x = tf.matmul(x, embeddings, transpose_b=transpose_embeddings) # (seqlen, batch * heads, seqlen)
        x = tf.reshape(x, (x.shape[0], -1, self.num_heads, x.shape[-1])) # (seqlen, batch, heads, seqlen)
        return tf.transpose(x, perm=[1, 2, 0, 3])

    def _compute_attn(self, query, value, key, mask):
        """
        :params:
            query, key, value: tensors of shape (..., seqlen, head_dims)
        :returns:
            attn_scores: Attention scores with shape (..., seqlen, head_dims), i.e., the attention scores
                for each head in each batch
            attn_weights: Attention weights applied to the value tensor
        """
        # technically, value may have a different dim, resulting in an attention shape of (..., seqlen, dim_v)
        # query and key must have the same dimensions
        key_dims = tf.shape(key)[-1]
        rpr_embeddings = self.rpr_embedding(self.rpr_lookup)

        alpha = tf.matmul(query, key, transpose_b=True) 
        alpha += self._compute_relative(query, rpr_embeddings, transpose_embeddings=True)
        alpha /= tf.sqrt(tf.cast(key_dims, tf.float32))

        if mask:
            alpha += mask * -np.inf
        attn_weights = tf.nn.softmax(alpha) # default last axis (key_dims)
        attn_scores = tf.matmul(attn_weights, value)
        attn_scores += self._compute_relative(attn_weights, rpr_embeddings, transpose_embeddings=False)

        return attn_scores, attn_weights

    def call(self, query, value, key=None, mask=None):
        """
        params:
            query, value, key: tensor of shape (batch, seqlen, dims)
        """
        if key is not None:
            key = value
        batch_size = tf.shape(query)[0] # or tf.shape(key)[0]

        # forward pass through weights and split (after for efficiency)
        query = self._project_to_heads(self.weights_q(query))
        value = self._project_to_heads(self.weights_v(value))
        key = self._project_to_heads(self.weights_k(key))

        # compute attention scores and grab weights
        attn_scores, attn_weights = self._compute_attn(query, value, key, mask)

        # transpose and reshape to concat scores for each head in each batch
        attn_scores = tf.transpose(attn_scores, perm=[0, 2, 1, 3]) # (batch_size, seqlen, heads, dims)
        concat_attn = tf.reshape(attn_scores, (batch_size, -1, self.dims)) # (batch_size, seqlen, dims)

        out = self.concat_head_weights(concat_attn)
        return out, attn_weights

    @classmethod
    def from_config(cls, **kwargs):
        pass

    def get_config(self):
        pass


# %%
_input = np.arange(20).reshape(2, 5, 2)
MultiHeadRelativeAttention(num_heads=2, dims=2, max_relative_pos=2, max_seqlen=5)(_input, _input, _input)

# %% [markdown]
# ## Encoder

# %%
class Encoder(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ffnn_dims, dropout_rate=0.2):
        super().__init__()
        self.attn = dict(
            layer=MultiHeadRelativeAttention(num_heads=num_heads, dims=embed_dim),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization())
        self.ffnn = dict(
            layer=keras.Sequential([
                keras.layers.Dense(ffnn_dims, activation="relu"),
                keras.layers.Dense(embed_dim)
            ]),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization())
    
    def call(self, inputs, training, padding_mask):
        attn_out = self.attn['layer'](query=inputs, key=inputs, value=inputs, mask=padding_mask)
        attn_out = self.attn['dropout'](attn_out, training=training)
        attn_out = self.attn['norm'](inputs + attn_out)

        ffnn_out = self.ffnn['layer'](attn_out)
        ffnn_out = self.ffnn['dropout'](ffnn_out, training=training)
        ffnn_out = self.ffnn['norm'](attn_out + ffnn_out)
        return ffnn_out

    def get_config(self):
        return {
            'attn': self.attn['layer'].get_config(),
            'attn_dropout': self.attn['dropout'].get_config(),
            'attn_norm': self.attn['norm'].get_config(),
            'ffnn': self.ffnn['layer'].get_config(),
            'ffnn_dropout': self.ffnn['dropout'].get_config(),
            'ffnn_norm': self.ffnn['norm'].get_config()
        }
    
    @classmethod
    def from_config(cls, **kwargs):
        cls.attn = dict(
            layer=MultiHeadRelativeAttention.from_config(kwargs['attn']),
            dropout=keras.layers.Dropout.from_config(kwargs['attn_dropout']),
            norm=keras.layers.LayerNormalization.from_config(kwargs['attn_norm'])
        )
        cls.ffnn = dict(
            layer=keras.Sequential.from_config(kwargs['ffnn']),
            dropout=keras.layers.Dropout.from_config(kwargs['ffnn_dropout']),
            norm=keras.layers.LayerNormalization.from_config(kwargs['ffnn_norm'])
        )
        return cls


# %%
class EncoderStack(keras.layers.Layer):
    def __init__(self, num_encoders, **kwargs):
        super().__init__()
        self.encoders = keras.Sequential([Encoder(**kwargs) for i in range(num_encoders)])

    def call(self, inputs, training, padding_mask):
        return self.encoders(inputs, training, padding_mask)
    
    def get_config(self):
        return {
            'encoders': self.encoders.get_config()
        }

    @classmethod
    def from_config(cls, **kwargs):
        cls.encoders = keras.Sequential.from_config(kwargs['encoders'])
        return cls

# %% [markdown]
# ## Decoder

# %%
class Decoder(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ffnn_dims, key_dim=None, dropout_rate=0.2):
        super().__init__()
        self.attn = dict(
            # use either specified key_dim (recommended key_dim < embed_dim) or just keep same size
            layer=MultiHeadRelativeAttention(num_heads=num_heads, key_dim=key_dim or embed_dim),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization()
        )
        self.encdec_attn = dict(
            layer=MultiHeadRelativeAttention(num_heads=num_heads, key_dim=embed_dim),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization()
        )
        self.ffnn = dict(
            layer=keras.Sequential([
                keras.layers.Dense(ffnn_dims, activation="relu"),
                keras.layers.Dense(embed_dim)
                ]),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization()
        )

    def call(self, inputs, training, padding_mask, lookahead_mask):
        # when q, k, and v are same, this is self attention
        # otherwise it is performing cross attention (across multiple documents)
        attn_out = self.attn['layer'](query=inputs, key=inputs, value=inputs, mask=lookahead_mask)
        attn_out = self.attn['dropout'](attn_out, training=training)
        attn_out = self.attn['norm'](inputs + attn_out)

        encdec_attn_out = self.encdec['layer'](query=attn_out, key=inputs, value=inputs, mask=padding_mask)
        encdec_attn_out = self.encdec['dropout'](encdec_attn_out, training=training)
        encdec_attn_out = self.encdec['norm'](attn_out + encdec_attn_out)

        ffnn_out = self.ffnn['layer'](encdec_attn_out)
        ffnn_out = self.ffnn['dropout'](ffnn_out, training=training)
        ffnn_out = self.ffnn['norm'](ffnn_out + encdec_attn_out)
        return ffnn_out

    def get_config(self):
        return {
            'attn': self.attn['layer'],
            'attn_dropout': self.attn['dropout'],
            'attn_norm': self.attn['norm'],
            'encdec_attn': self.encdec_attn['layer'],
            'encdec_attn_dropout': self.encdec_attn['dropout'],
            'encdec_attn_norm': self.encdec_attn['norm'],
            'ffnn': self.ffnn['layer'],
            'ffnn_dropout': self.ffnn['dropout'],
            'ffnn_norm': self.ffnn['norm']
        }

    @classmethod
    def from_config(cls, **kwargs):
        cls.attn = dict(
            layer=MultiHeadRelativeAttention.from_config(kwargs['attn']),
            dropout=keras.layers.Dropout.from_config(kwargs['attn_dropout']),
            norm=keras.layers.LayerNormalization.from_config(kwargs['attn_norm'])
        )
        cls.encdec_attn = dict(
            layer=MultiHeadRelativeAttention.from_config(kwargs['encdec_attn']),
            dropout = keras.layers.Dropout.from_config(kwargs['encdec_attn_dropout']),
            norm = keras.layers.LayerNormalization.from_config(kwargs['encdec_attn_norm'])
        )
        cls.ffnn = dict(
            layer=keras.Sequential.from_config(kwargs['ffnn']),
            dropout=keras.layers.Dropout.from_config(kwargs['ffnn_dropout']),
            norm=keras.layers.LayerNormalization.from_config(kwargs['ffnn_norm'])
        )
        return cls


# %%
class DecoderStack(keras.layers.Layer):
    def __init__(self, num_decoders, **kwargs):
        super().__init__()
        self.decoders = keras.Sequential([Decoder(**kwargs) for i in range(num_decoders)])

    def call(self, inputs, training, padding_mask, lookahead_mask):
        return self.decoders(inputs, training, padding_mask, lookahead_mask)

    def get_config(self):
        return {
            'decoders': self.decoders.from_config()
        }

    @classmethod
    def from_config(cls, **kwargs):
        cls.decoders = keras.Sequential.from_config(kwargs['decoders'])
        return cls

# %% [markdown]
# ## Embeddings
# %% [markdown]
# ### Positional Embeddings
# 
# %% [markdown]
# $$\large{\overrightarrow{p_t}^{(i)} = \begin{cases} sin(\omega(k) * t), & \mbox{if  } i = 2 * k \\ cos(\omega(k) * t), & \mbox{if  } i = 2 * k + 1 \end{cases}}$$
# %% [markdown]
# $$\large{\omega(k) = \frac{1}{10000^{2k / d}}}$$

# %%
OMEGA_SCALE = 10000

def omega(k, dims):
    return 1 / (np.power(OMEGA_SCALE, 2 * k / np.float64(dims)))

def positional_embeddings(max_pos, dims):
    """
    Returns a 2-d numpy array with shape (max_pos, dims) representing positional embeddings up to `max_pos`
    and compressed into `dims` dimensions
    """
    positions = np.arange(max_pos)[:, np.newaxis]
    embeddings = positions * omega(np.arange(dims, dtype=int) // 2, dims)
    embeddings[:, 0::2] = np.sin(embeddings[:, 0::2])
    embeddings[:, 1::2] = np.cos(embeddings[:, 1::2])
    return embeddings


# %%
EMBED_DIMS = 64
posembed = positional_embeddings(100, EMBED_DIMS)

fig, ax = plt.subplots(figsize=(5,5))
colormap = ax.pcolormesh(posembed, cmap="RdBu")
ax.set_xlim((0, EMBED_DIMS))
fig.colorbar(colormap, ax=ax)

plt.show()

# %% [markdown]
# ### Embedding Layer

# %%
class Embedding(keras.layers.Layer):
    def __init__(self, vocab_size, dims, max_seqlen, dropout_rate=0.2):
        super().__init__()
        self.token_embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=dims)
        self.positional_embedding = positional_embeddings(max_seqlen, dims)
        self.dropout = keras.layers.Dropout(rate=dropout_rate)

        self.max_seqlen = max_seqlen
        self.dims = dims

    def call(self, inputs):
        max_seqlen_in_batch = np.shape(inputs)[-1]
        embed_out = self.token_embedding(inputs)
        # embed_out 3d (batch_size, max_seqlen_in_batch, dims) 
        embed_out += self.positional_embedding[:max_seqlen_in_batch] 
        # positional_embedding 2d (max_seqlen, dims) compressed to (max_seqlen_in_batch, dims)
        # then broadcasted to (batch_size, max_seqlen_in_batch, dims), summed with embed_out
        embed_out = self.dropout(embed_out)
        return embed_out

    def get_config(self):
        return {
            'token': self.token_embedding.get_config(),
            'max_seqlen': self.max_seqlen,
            'dims': self.dims,
            'dropout': self.dropout.get_config()
        }

    @classmethod
    def from_config(cls, **kwargs):
        cls.token_embedding = keras.layers.Embedding.from_config(kwargs['token'])
        cls.positional_embedding = kwargs['positional']
        cls.dims = kwargs['dims']
        cls.dropout = keras.layers.Dropout.from_config(kwargs['dropout'])
        return cls

# %% [markdown]
# ### Adjusted Positional Encoding
# 
# Because some notes will be played at the same time, these notes must have an embedding that represents that they're played at the same time. I could give these notes the same positional encoding, but this muddies the time relationship between notes, and disregards the fact that notes will be read/generated sequentially, even in a chord. 
# 
# I'll **keep the standard positional encoding**, but also add another embedding: **a chord embedding**. This **chord embedding** will use the relative position mechanism discovered by Shaw, Uszkoreit, Vaswani as inspiration. 
# %% [markdown]
# ## Regularization 
# Use l2 MaxNorm weight constraints, Early Stopping, Adam's Optimizer LR Schedule, Dropout
# %% [markdown]
# ## Hyperparameters
# * Specific to Model
#     * FFNN Dimensions for both encoder and decoder
#     * Number of Heads
#     * Embedding Dimensions
#     * Num Encoders + Num Decoders (set max limit)
#     * Key dimensions
#     * Dropout Rate
# * General
#     * Learning Rate
#     * Adam's Optimizer
#         * b1 decay
#         * b2 decay
#         * alpha
