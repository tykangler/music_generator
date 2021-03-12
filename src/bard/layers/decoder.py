import tensorflow as tf
from tensorflow import keras
import numpy as np
from .attention import *

class Decoder(keras.layers.Layer):
   """
   A single decoder layer to be used in the decoder stack. Computes self attention for the output sequence and cross attention
   with the output of the encoder stack. Outputs cross attention scores with encoder output as the K, V matrices, and decoder
   self attention output as the Q matrix. Given a sequence with shape (batch, q_seqlen, dim), the output shape will be 
   (batch, q_seqlen, dim)
   """
   def __init__(self, heads, max_relative_pos, ffnn_dim, dropout_rate=0.2, kernel_constraint=None, **kwargs):
      super().__init__(**kwargs)
      self.heads = heads
      self.max_relative_pos = max_relative_pos
      self.ffnn_dim = ffnn_dim
      self.dropout_rate = dropout_rate
      self.kernel_constraint = kernel_constraint
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
               keras.layers.Dense(self.ffnn_dim, kernel_constraint=self.kernel_constraint, activation="relu"),
               keras.layers.Dense(embed_dim, kernel_constraint=self.kernel_constraint)
         ], name="Decoder Pointwise Feed Forward"),
         dropout=keras.layers.Dropout(rate=self.dropout_rate),
         norm=keras.layers.LayerNormalization()
      )
      super().build(input_shape)

   def call(self, inputs, enc_kv, padding_mask, lookahead_mask, training):
      """
      performs a forward pass.
      params:
         inputs: tensor of shape (batch, q_seqlen, dim)
         env_kv: encoder output. tensor of shape (batch, seqlen, dim)
         padding_mask: tensor of shape (batch, q_seqlen, seqlen) used in the cross attention step 
               to mask encoder padding
         lookahead_mask: tensor of shape (batch, q_seqlen, q_seqlen) used in the self attention step to 
               mask future positions, as well as decoder padding
         training: boolean representing whether this is a train step
      """
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
         dropout_rate=self.dropout_rate,
         kernel_constraint=self.kernel_constraint.get_config()
      ))
      return config

   @classmethod
   def from_config(cls, **kwargs):
      return cls(**kwargs)

class DecoderStack(keras.layers.Layer):
   """
   Stack of decoder layers. Output of the ith decoder is passed into the i+1th decoder sequentially until reaching 
   `units` decoders. All params besides `units` are forwarded into each decoder identically. 
   """
   def __init__(self, units, heads, ffnn_dim, max_relative_pos, dropout_rate=0.2, kernel_constraint=None, **kwargs):
      super().__init__(**kwargs)
      self.units = units
      self.heads = heads
      self.ffnn_dim = ffnn_dim
      self.max_relative_pos = max_relative_pos
      self.dropout_rate = dropout_rate
      self.kernel_constraint = kernel_constraint
      self.decoders = [Decoder(self.heads, self.ffnn_dim, self.max_relative_pos, self.dropout_rate, self.kernel_constraint)
                        for i in range(self.units)]

   def call(self, inputs, enc_kv, padding_mask, lookahead_mask, training):
      """
      forward pass through all decoders in this stack.
      params:
         inputs: tensor of shape (batch, q_seqlen, dim)
         enc_kv: tensor of shape (batch, seqlen, dim)
         padding_mask: tensor of shape (batch, q_seqlen, seqlen) used to mask encoder padding
         lookahead_mask: tensor of shape (batch, q_seqlen, q_seqlen) used in self attention step to
               mask future positions and decoder padding
         training: boolean representing whether is part of a train step
      """
      all_attn_weights = dict()
      for i, layer in enumerate(self.decoders):
         inputs, *attn_weights = layer(inputs, enc_kv, padding_mask, lookahead_mask, training)
         all_attn_weights[f'decoder_{i}'] = attn_weights
      return inputs, all_attn_weights

   def get_config(self):
      config = super().get_config()
      config.update(dict(
         units=self.units,
         embed_dim=self.embed_dim,
         heads=self.heads,
         ffnn_dim=self.ffnn_dim,
         dropout_rate=self.dropout_rate,
         kernel_constraint=self.kernel_constraint
      ))
      return config

   @classmethod
   def from_config(cls, **kwargs):
      return cls(**kwargs)
