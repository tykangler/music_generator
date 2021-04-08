import tensorflow as tf
from tensorflow import keras

from .attention import *


class Encoder(keras.layers.Layer):
   """
   A single encoder layer to be used in the encoder stack. This takes in an embedded input sequence, calculates self 
   attention weights, and outputs self attention scores for the embedded input sequence.

   params:
      heads: the number of heads to split the input sequence into in the relative attention step.
      ffnn_dim: the number of hidden layer units in the position wise feed forward step.
      max_relative_pos: the max distance considered for relative attention
      dropout_rate: rate of dropout 
      kernel_constraint: weight constraints for the attention and ffnn step
   """
   def __init__(self, heads, ffnn_dim, max_relative_pos, dropout_rate=0.2, kernel_constraint=None, **kwargs):
      super().__init__(**kwargs)
      self.heads = heads
      self.ffnn_dim = ffnn_dim
      self.dropout_rate = dropout_rate
      self.kernel_constraint = keras.constraints.get(kernel_constraint)
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
               keras.layers.Dense(self.ffnn_dim, kernel_constraint=self.kernel_constraint, activation="relu"),
               keras.layers.Dense(embed_dim, kernel_constraint=self.kernel_constraint)
         ], name="Encoder Pointwise Feed Forward"),
         dropout=keras.layers.Dropout(rate=self.dropout_rate),
         norm=keras.layers.LayerNormalization())
      super().build(input_shape)
   
   def call(self, inputs, padding_mask, training):
      """
      computes self attention, passes result through a ffnn with one hidden layer, and employs residual
      connections through a layer norm
      params:
         inputs: tensor of shape (batch, q_seqlen, dim)
         padding_mask: tensor of shape (batch, q_seqlen, q_seqlen)
         training: boolean representing whether this is a train step
      returns:
         ffnn_out: tensor of shape (batch, q_seqlen, dim)
         attn_weights: tensor of shape (batch, heads, q_seqlen, q_seqlen) representing weight mappings from each
               word to each other word in the input sequence for each attention head
      """
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
         dropout_rate=self.dropout_rate,
         kernel_constraint=self.kernel_constraint.get_config()
      ))
      return config
   
   @classmethod
   def from_config(cls, **kwargs):
      return NotImplemented


class EncoderStack(keras.layers.Layer):
   """
   Aggregates `units` number of Encoders sequentially. The output of an encoder is passed into another encoder `units` times.
   All other parameters are forwarded into each encoder identically.
   params:
      units: the number of encoders
   """
   def __init__(self, units, heads, ffnn_dim, max_relative_pos, dropout_rate=0.2, kernel_constraint=None, **kwargs):
      super().__init__(**kwargs)
      self.units = units
      self.heads = heads
      self.ffnn_dim = ffnn_dim
      self.dropout_rate = dropout_rate
      self.max_relative_pos = max_relative_pos
      self.kernel_constraint = kernel_constraint
      self.encoders = [Encoder(self.heads, self.ffnn_dim, self.max_relative_pos, self.dropout_rate, self.kernel_constraint) 
                        for i in range(self.units)]
   
   def call(self, inputs, padding_mask, training):
      """
      runs forward pass through each encoder in this stack sequentially. Each encoder's attention weights are 
      returned as a dictionary, mapping layer to its respective weights.
      params:
         inputs: tensor of shape (batch, q_seqlen, dims)
         padding_mask: tensor of shape (batch, q_seqlen, q_seqlen)
         training: boolean representing whether this is a train step
      returns:
         tensor of shape (batch, q_seqlen, dims)
         dictionary of attention weights, with "encoder_{i}" as keys            
      """
      all_attn_weights = dict()
      for i, layer in enumerate(self.encoders):
         inputs, attn_weights = layer(inputs, padding_mask, training)
         all_attn_weights[f'encoder_{i}'] = attn_weights
      return inputs, all_attn_weights
   
   def get_config(self):
      config = super().get_config()
      config.update(dict(
         units=self.units,
         heads=self.heads,
         ffnn_dim=self.ffnn_dim,
         dropout_rate=self.dropout_rate,
         kernel_constraint=keras.constraints.serialize(self.kernel_constraint)
      ))
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)
