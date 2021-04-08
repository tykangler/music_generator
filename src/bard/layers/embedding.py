import numpy as np
import tensorflow as tf
from tensorflow import keras

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


class Embedding(keras.layers.Layer):
   """
   Embeds sequence of integer labels into vectors of fixed size. Each element in each batch
   is converted to a vector of size `output_dim`.
   params:
      vocab_size: size of the vocabulary, all integers should be within the vocabulary 
      output_dim: dimensions of the output vectors
      dropout_rate: dropout rate
      embeddings_constraint: weight constraint for embeddings
   """
   def __init__(self, vocab_size, output_dim, dropout_rate=0.2, embeddings_constraint=None, **kwargs):
      super().__init__(**kwargs)
      self.vocab_size = vocab_size
      self.output_dim = output_dim
      self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
      self.token_embedding = keras.layers.Embedding(
         input_dim=vocab_size, 
         output_dim=output_dim,
         embeddings_constraint=self.embeddings_constraint,
         name="token embedding")
      self.dropout = keras.layers.Dropout(rate=dropout_rate)

   def build(self, input_shape):
      seqlen = input_shape[-1]
      self.positional_embedding = positional_embeddings(seqlen, self.output_dim)
      super().build(input_shape)

   def call(self, inputs, training):
      """
      params:
         inputs: tensor of shape (batch, seqlen)
         training: boolean representing whether this is a train step
      returns:
         tensor of shape (batch, seqlen, output_dim)
      """
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
         dropout_rate=self.dropout_rate,
         embeddings_constraint=keras.constraints.serialize(self.embeddings_constraint)
      ))
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)
