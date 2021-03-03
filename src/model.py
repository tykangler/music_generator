#!/usr/bin/env python
# coding: utf-8

# # Model Architecture
# 
# 1276 total files, 962 train files (75%)

# In[1]:


import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import collections


# ## Multi-Head Relative Attention

# In[14]:


class MultiHeadRelativeAttention(keras.layers.Layer):
    """
    An implementation of the multi head attention mechanism, with relative position representations. 
    The input K, and V matrices have the shape of (batch, seqlen, dim). The input Q matrix has the shape (batch, query_seqlen, dim).
    The output shape will equal the shape of the query matrix. 
    :params:
        heads: the number of heads to project q, k, and v matrices to
        dim: the dimensions of the weighted query and key matrices
        max_relative_pos: the 
    """
    def __init__(self, heads, max_relative_pos, key_dim=None, value_dim=None, **kwargs):
        super().__init__(**kwargs)
        # query and key will have the same dimensions. value may or may not have the same dimensions. if value 
        # not specified, then value = key
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_relative_pos = max_relative_pos

    def build(self, input_shape):
        batch, query_seqlen, dim_input = input_shape

        # dims calculation
        qk_dim = self.key_dim or dim_input
        v_dim = self.value_dim or dim_input
        assert qk_dim % self.heads == 0, """q, k dims must be a multiple of heads"""
        assert v_dim % self.heads == 0, """v dims must be a multiple of heads"""
        self.head_qk_dim = qk_dim // self.heads
        self.head_v_dim = v_dim // self.heads
 
        # relative positional encoding
        num_rprs = self.max_relative_pos * 2 + 1
        self.rpr_key_embedding = keras.layers.Embedding(num_rprs, self.head_qk_dim, name="relative embedding (key)")
        self.rpr_value_embedding = keras.layers.Embedding(num_rprs, self.head_v_dim, name="relative embedding (value)")
        self.rpr_lookup = self._generate_rpr_lookup(query_seqlen, self.max_relative_pos)

        # project to heads after applying weights/dense
        self.weights_q = keras.layers.Dense(qk_dim, use_bias=False, name="query weights")
        self.weights_k = keras.layers.Dense(qk_dim, use_bias=False, name="key weights")
        self.weights_v = keras.layers.Dense(v_dim, use_bias=False, name="value weights")

        # concatenated heads passed as input
        self.concat_head_weights = keras.layers.Dense(input_shape[-1], name="concat weights")

        super().build(input_shape)

    def _generate_rpr_lookup(self, query_seqlen, max_relative_pos):
        x = np.arange(query_seqlen)
        x = tf.expand_dims(x, axis=0) - tf.expand_dims(x, axis=1)
        x = tf.clip_by_value(x, -max_relative_pos, max_relative_pos)
        return x + max_relative_pos

    def call(self, query, value, key=None, mask=None):
        """
        params:
            query: tensor of shape (batch, q_seqlen, dim)
            key: tensor of shape (batch, seqlen, dim)
            value: tensor of shape (batch, seqlen, dim)
            mask: tensor with shape equal to or broadcastable to (batch, q_seqlen, seqlen)
        :returns:
            attn_scores: Attention scores with shape (..., q_seqlen, dim), i.e., the attention scores
                for each head in each batch
            attn_weights: Attention weights applied to the value tensor
        """
        if key is not None:
            key = value
        batch_size = tf.shape(query)[0] # or tf.shape(key)[0]
        dim_input = tf.shape(query)[-1]

        # forward pass through weights and split (after for efficiency)
        query = self._project_to_heads(self.weights_q(query))
        key = self._project_to_heads(self.weights_k(key))
        value = self._project_to_heads(self.weights_v(value))   

        # compute attention scores and grab weights
        attn_scores, attn_weights = self._compute_attn(query, value, key, mask)

        # transpose and reshape to concat scores for each head in each batch
        attn_scores = tf.transpose(attn_scores, perm=[0, 2, 1, 3]) # (batch_size, seqlen, heads, head_dim)
        concat_attn = tf.reshape(attn_scores, (batch_size, -1, dim_input)) # (batch_size, seqlen, dim)

        out = self.concat_head_weights(concat_attn)
        return out, attn_weights

    def _project_to_heads(self, x):
        """
        projects x to `heads` number of heads
        params:
            x: tensor of shape (batch, seqlen, dim)
        returns:
            tensor of shape (batch, heads, seqlen, head_dim). Each sequence in the batch is 
                split column wise so that each head attends to a subsection of dim in each sequence.
        example:
            x with shape (batch=2, seqlen=3, dim=4)
            [[[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]],

             [[12, 13, 14, 15],
              [16, 17, 18, 19],
              [20, 21, 22, 23]]]
            reshape to (batch=2, seqlen=3, heads=2, head_dim=2)
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
            transpose to (batch=2, heads=2, seqlen=3, head_dim=2)
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
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.heads, -1))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _compute_relative(self, x, embeddings, transpose_embeddings=False):
        """
        computes relational attention across all input sequences in heads and batches. x has 
        input shape (batch, heads, seqlen, head_dim) where head_dim refers to either head_qk_dim or head_v_dim.
        
        For query, `x` is transposed to (q_seqlen, batch, heads, head_dim), reshaped to 
        (q_seqlen, batch * heads, head_dim), then multiplied with embeddings with shape 
        (q_seqlen, head_dim, q_seqlen) (after transpose). The resulting shape is 
        (q_seqlen, batch * heads, q_seqlen). 
        
        For attn_weights, `x` is transposed to (q_seqlen, batch, heads, q_seqlen), reshaped to 
        (q_seqlen, batch * heads, q_seqlen), then multiplied with embeddings with shape 
        (q_seqlen, q_seqlen, head_dim). The resulting shape is (q_seqlen, batch * heads, head_dim). 
        In both cases, the result wil be reshaped back to (q_seqlen, batch, heads, ...), then transposed back to 
        (batch, heads, q_seqlen, ...). This approach avoids broadcasting.
        params:
            x: tensor (query or attn weights) with shape (batch, heads, q_seqlen, head_dim)
            embeddings: learned rpr embeddings with shape (q_seqlen, q_seqlen, head_dim)
            transpose_embeddings: Whether to transpose the embeddings argument, pass true for query 
        """
        x = tf.transpose(x, perm=[2, 0, 1, 3]) # (q_seqlen, batch, heads, head_dim)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1])) # (q_seqlen, batch * heads, head_dim)
        x = tf.matmul(x, embeddings, transpose_b=transpose_embeddings) # (q_seqlen, batch * heads, q_seqlen)
        x = tf.reshape(x, (x.shape[0], -1, self.heads, x.shape[-1])) # (q_seqlen, batch, heads, q_seqlen)
        return tf.transpose(x, perm=[1, 2, 0, 3])

    def _compute_attn(self, query, value, key, mask):
        """

        :params:
            query: tensors of shape (batch, heads, q_seqlen, head_qk_dim)
            value: tensors of shape (batch, heads, seqlen, head_v_dim)
            key: tensors of shape (batch, heads, seqlen, head_qk_dim)
            mask: tensor of shape (batch, q_seqlen, seqlen) applied after first matmul and just prior to softmax
        :returns:
            attn_scores: Attention scores with shape (..., seqlen, dim), i.e., the attention scores
                for each head in each batch
            attn_weights: Attention weights applied to the value tensor
        """
        # technically, value may have a different dim, resulting in an attention shape of (..., seqlen, dim_v)
        # query and key must have the same dimensions
        key_dims = tf.shape(key)[-1]

        alpha = tf.matmul(query, key, transpose_b=True) 
        rpr_key_embedding = self.rpr_key_embedding(self.rpr_lookup)
        alpha += self._compute_relative(query, rpr_key_embedding, transpose_embeddings=True)
        alpha /= tf.sqrt(tf.cast(key_dims, tf.float32))

        if mask:
            alpha += mask[:, tf.newaxis, :, :] * -np.inf
        attn_weights = tf.nn.softmax(alpha) # default last axis (key_dims)
        attn_scores = tf.matmul(attn_weights, value)
        rpr_value_embedding = self.rpr_value_embedding(self.rpr_lookup)
        attn_scores += self._compute_relative(attn_weights, rpr_value_embedding, transpose_embeddings=False)

        return attn_scores, attn_weights

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            heads=self.heads,
            max_relative_pos=self.max_relative_pos,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
        ))
        return config


# ## Encoder

# In[3]:


class Encoder(keras.layers.Layer):
    def __init__(self, heads, ffnn_dim, max_relative_pos, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.ffnn_dim = ffnn_dim
        self.dropout_rate = dropout_rate
        self.attn = dict(
            layer=MultiHeadRelativeAttention(
                heads=heads, 
                max_relative_pos=max_relative_pos, 
                name="Encoder Relative Attention"),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization())

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.ffnn = dict(
            layer=keras.Sequential([
                keras.layers.Dense(self.ffnn_dim, activation="relu"),
                keras.layers.Dense(embed_dim)
            ], name="Encoder Pointwise Feed Forward"),
            dropout=keras.layers.Dropout(rate=self.dropout_rate),
            norm=keras.layers.LayerNormalization())
        super().build(input_shape)
    
    def call(self, inputs, padding_mask, training):
        attn_out, attn_weights = self.attn['layer'](query=inputs, key=inputs, value=inputs, mask=padding_mask)
        attn_out = self.attn['dropout'](attn_out, training=training)
        attn_out = self.attn['norm'](inputs + attn_out)

        ffnn_out = self.ffnn['layer'](attn_out)
        ffnn_out = self.ffnn['dropout'](ffnn_out, training=training)
        ffnn_out = self.ffnn['norm'](attn_out + ffnn_out)
        return ffnn_out, attn_weights

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            embed_dim=self.embed_dim,
            heads=self.heads,
            ffnn_dim=self.ffnn_dim,
            dropout_rate=self.dropout_rate
        ))
        return config
    
    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)


# In[4]:


class EncoderStack(keras.layers.Layer):
    def __init__(self, units, heads, ffnn_dim, max_relative_pos, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.heads = heads
        self.ffnn_dim = ffnn_dim
        self.dropout_rate = dropout_rate
        self.max_relative_pos = max_relative_pos
        self.encoders = [Encoder(self.heads, self.ffnn_dim, self.max_relative_pos, self.dropout_rate) 
                         for i in range(self.units)]
    
    def call(self, inputs, padding_mask, training):
        all_attn_weights = dict()
        for i, layer in enumerate(self.encoders):
            inputs, attn_weights = layer(inputs, padding_mask, training)
            all_attn_weights[f'encoder{i}'] = attn_weights
        return inputs, all_attn_weights
    
    def get_config(self):
        config = super().get_config()
        config.update(dict(
            units=self.units,
            heads=self.heads,
            ffnn_dim=self.ffnn_dim,
            dropout_rate=self.dropout_rate
        ))
        return config

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)


# ## Decoder

# In[5]:


class Decoder(keras.layers.Layer):
    def __init__(self, heads, max_relative_pos, ffnn_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.max_relative_pos = max_relative_pos
        self.ffnn_dim = ffnn_dim
        self.dropout_rate = dropout_rate
        self.attn = dict(
            # use either specified key_dim (recommended key_dim < embed_dim) or just keep same size
            layer=MultiHeadRelativeAttention(
                heads=heads, 
                max_relative_pos=max_relative_pos,
                name="Decoder Relative Attention"),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization()
        )
        self.encdec_attn = dict(
            layer=MultiHeadRelativeAttention(
                heads=heads, 
                max_relative_pos=max_relative_pos, 
                name="Encoder-Decoder Relative Attention"),
            dropout=keras.layers.Dropout(rate=dropout_rate),
            norm=keras.layers.LayerNormalization()
        )

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.ffnn = dict(
            layer=keras.Sequential([
                keras.layers.Dense(self.ffnn_dim, activation="relu"),
                keras.layers.Dense(embed_dim, activation="relu")
            ], name="Decoder Pointwise Feed Forward"),
            dropout=keras.layers.Dropout(rate=self.dropout_rate),
            norm=keras.layers.LayerNormalization()
        )
        super().build(input_shape)

    def call(self, inputs, enc_kv, padding_mask, lookahead_mask, training):
        # when q, k, and v are same, this is self attention
        # otherwise it is performing cross attention
        attn_out, attn_weights = self.attn['layer'](query=inputs, key=inputs, value=inputs, mask=lookahead_mask)
        attn_out = self.attn['dropout'](attn_out, training=training)
        attn_out = self.attn['norm'](inputs + attn_out)

        encdec_attn_out, encdec_attn_weights = self.encdec['layer'](query=attn_out, key=enc_kv, value=enc_kv, mask=padding_mask)
        encdec_attn_out = self.encdec['dropout'](encdec_attn_out, training=training)
        encdec_attn_out = self.encdec['norm'](attn_out + encdec_attn_out)

        ffnn_out = self.ffnn['layer'](encdec_attn_out)
        ffnn_out = self.ffnn['dropout'](ffnn_out, training=training)
        ffnn_out = self.ffnn['norm'](ffnn_out + encdec_attn_out)
        return ffnn_out, attn_weights, encdec_attn_weights

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            embed_dim=self.embed_dim,
            heads=self.heads,
            ffnn_dim=self.ffnn_dim,
            dropout_rate=self.dropout_rate
        ))
        return config

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)


# In[6]:


class DecoderStack(keras.layers.Layer):
    def __init__(self, units, heads, ffnn_dim, max_relative_pos, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.heads = heads
        self.ffnn_dim = ffnn_dim
        self.max_relative_pos = max_relative_pos
        self.dropout_rate = dropout_rate
        self.decoders = [Decoder(self.heads, self.ffnn_dim, self.max_relative_pos, self.dropout_rate)
                         for i in range(self.units)]

    def call(self, inputs, enc_kv, padding_mask, lookahead_mask, training):
        all_attn_weights = dict()
        DecoderAttentionWeights = collections.namedtuple('DecoderAttentionWeights', ['self_attn', 'cross_attn'])
        for i, layer in enumerate(self.decoders):
            inputs, *attn_weights = layer(inputs, enc_kv, padding_mask, lookahead_mask, training)
            all_attn_weights[f'decoder{i}'] = DecoderAttentionWeights(*attn_weights)
        return inputs, all_attn_weights

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            units=self.units,
            embed_dim=self.embed_dim,
            heads=self.heads,
            ffnn_dim=self.ffnn_dim,
            dropout_rate=self.dropout_rate
        ))
        return config

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)


# ## Embeddings

# ### Positional Embeddings
# 

# $$\large{\overrightarrow{p_t}^{(i)} = \begin{cases} sin(\omega(k) * t), & \mbox{if  } i = 2 * k \\ cos(\omega(k) * t), & \mbox{if  } i = 2 * k + 1 \end{cases}}$$

# $$\large{\omega(k) = \frac{1}{10000^{2k / d}}}$$

# In[7]:


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


# In[8]:


EMBED_DIMS = 64
posembed = positional_embeddings(100, EMBED_DIMS)

fig, ax = plt.subplots(figsize=(5,5))
colormap = ax.pcolormesh(posembed, cmap="RdBu")
ax.set_xlim((0, EMBED_DIMS))
fig.colorbar(colormap, ax=ax)

plt.show()


# ### Embedding Layer

# In[9]:


class Embedding(keras.layers.Layer):
    def __init__(self, vocab_size, output_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=output_dim,
            name="token embedding")
        self.dropout = keras.layers.Dropout(rate=dropout_rate)

    def build(self, input_shape):
        seqlen = input_shape[-1]
        self.positional_embedding = positional_embeddings(seqlen, self.output_dim)
        super().build(input_shape)

    def call(self, inputs, training):
        max_seqlen_in_batch = np.shape(inputs)[-1]
        embed_out = self.token_embedding(inputs)

        # embed_out 3d (batch_size, max_seqlen_in_batch, output_dim) 
        embed_out += self.positional_embedding[:max_seqlen_in_batch]

        # positional_embedding 2d (seqlen, output_dim) compressed to (max_seqlen_in_batch, output_dim)
        # then broadcasted to (batch_size, max_seqlen_in_batch, output_dim), summed with embed_out
        embed_out = self.dropout(embed_out, training=training)

        return embed_out

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            vocab_size=self.vocab_size,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate
        ))
        return config

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)


# ## Masking

# In[15]:


def create_padding_mask(inputs, pad_value=0):
    """
    returns tensor with same shape as inputs. The result will have 1's where elements equal pad_value,
    and 0's otherwise.
    params:
        inputs: the input tensor
        pad_value: padding token to compare equality against
    """
    mask = tf.cast(tf.math.equal(inputs, pad_value), tf.float32)
    return mask 


# In[20]:


def create_lookahead_mask(dim):
    """
    returns tensor with shape of (dim, dim) where the upper right triangle is populated with 1's and the rest 
    populated with 0. 

    mask[i, j] = 1 if j - i > 0 else 0
    """
    mask  = 1 - tf.linalg.band_part(tf.ones((dim, dim)), -1, 0)
    return mask


# In[21]:


def create_decoder_mask(inputs, pad_value=0):
    """
    returns tensor with shape (batch, seqlen, seqlen) given input tensor has shape (batch, seqlen). Combines
    the pad mask and lookahead mask for this batch of input sequences.
    """
    pad_mask = create_padding_mask(inputs, pad_value) # (batch, seqlen)
    lookahead_mask = create_lookahead_mask(inputs.shape[1]) # (seqlen, seqlen)
    pad_mask = pad_mask[:, tf.newaxis, :] # (batch, 1, seqlen)
    return tf.maximum(pad_mask, lookahead_mask)
    


# ## Relative Embedding
# 
# Because some notes will be played at the same time, these notes must have an embedding that represents that they're played at the same time. I could give these notes the same positional encoding, but this muddies the time relationship between notes, and disregards the fact that notes will be read/generated sequentially, even in a chord. 
# 
# I'll **keep the standard positional encoding**, but also add another embedding: **a chord embedding**. This **chord embedding** will use the relative position mechanism created by Shaw, Uszkoreit, Vaswani as inspiration. 

# ## Model

# In[12]:


class Transfomer(keras.Model):
    def __init__(self, vocab_size, embed_dim, layers, heads, 
                 key_dim, value_dim, ffnn_dim, max_relative_pos, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.layers = layers
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.ffnn_dim = ffnn_dim
        self.max_relative_pos = max_relative_pos
        self.dropout_rate = dropout_rate
        self.encoders = EncoderStack(
            units=layers, 
            heads=heads, 
            ffnn_dim=ffnn_dim, 
            max_relative_pos=max_relative_pos,
            dropout_rate=dropout_rate,
            name=f"Encoder Stack ({layers} layers)")
        self.decoders = DecoderStack(
            units=layers,
            heads=heads,
            ffnn_dim=ffnn_dim,
            max_relative_pos=max_relative_pos,
            dropout_rate=dropout_rate,
            name=f"Decoder Stack ({layers} layers)"
        )
        self.enc_embedding = Embedding(
            output_dim=embed_dim,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
            name="Encoder Embedding"
        )
        self.dec_embedding = Embedding(
            output_dim=embed_dim,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
            name="Decoder Embedding"
        )
        self.linear = keras.layers.Dense(vocab_size, activation=keras.activations.softmax, name="Dense Layer + Softmax")

    def call(self, inputs, targets, padding_mask, lookahead_mask, training):
        enc_out = self.enc_embedding(inputs=inputs, training=training)
        enc_out, encoder_attn_weights = self.encoders(inputs=enc_out, padding_mask=padding_mask, training=training)
        dec_out = self.dec_embedding(inputs=targets, training=training)
        dec_out, dec_weights = self.decoders(
            inputs=dec_out, 
            enc_kv=enc_out, 
            padding_mask=padding_mask, # to mask encoder output where padding exists
            lookahead_mask=lookahead_mask, 
            training=training)
        result, decoder_attn_weights = self.linear(dec_out)
        return result, encoder_attn_weights, decoder_attn_weights
    
    def train_step(self, data):
        inputs, targets = data # ([batch, seqlen], [batch, seqlen + 2])
        # ['start', ...], [..., 'end']
        target_input, target_output = targets[:, :-1], targets[:, 1:]

        # dec_mask (used in self attn block) combines lookahead_mask with dec_key_mask, 
        # while dec_pad_mask (used in cross attn block) masks enc kv padding
        pad_mask = create_padding_mask(inputs) # (batch, seqlen)
        dec_mask = create_decoder_mask(targets) # (batch, q_seqlen, seqlen) q_seqlen == seqlen
        
        # mask key portions, the innermost dimension
        # q * k^T has shape (batch, heads, q_seqlen, seqlen), so mask seqlen
        # shape of mask should be (batch, q_seqlen, seqlen) masking seqlen
        pad_mask = pad_mask[:, tf.newaxis, :] # (batch, q_seqlen=1 [broadcast], seqlen)

        with tf.GradientTape() as tape:
            y_pred = self(x, targets_input, )


# ## Regularization 
# Use l2 MaxNorm weight constraints, Early Stopping, Adam's Optimizer LR Schedule, Dropout

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
