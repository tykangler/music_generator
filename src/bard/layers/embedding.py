import numpy as np
import tensorflow as tf
from tensorflow import keras

from .utils import underlying_value

# consider using this for positional embeddings to interleave tensors by even and odd indices.
# not currently used
# example usage:
#     complete_positional_embeddings = alternate_by_even_odd(
#        np.sin(embeddings[:, 0::2]), np.cos(embeddings[:, 1::2]))
def alternate_by_even_odd(even_index_tensor, odd_index_tensor):
   assert tf.reduce_all(tf.shape(even_index_tensor) == tf.shape(odd_index_tensor))
   concat = tf.concat([even_index_tensor[..., tf.newaxis],
                        odd_index_tensor[..., tf.newaxis]], axis=-1)
   # even = [1, 2, 3], odd = [4, 5, 6]
   # even = [[1], [2], [3]], odd = [[4], [5], [6]]
   # concat = [[1, 4],
   #           [2, 5],
   #           [3, 6]]
   return tf.reshape(concat, shape=[-1, tf.shape(even_index_tensor)[-1] * 2])

OMEGA_SCALE = 10000

def omega(k, dims):
   return 1 / (np.power(OMEGA_SCALE, 2 * k / np.float32(dims)))

def positional_embeddings(max_pos, dims):
   """
   Returns a 2-d numpy array with shape (max_pos, dims) representing positional embeddings up to `max_pos`
   and compressed into `dims` dimensions
   """
   positions = np.arange(max_pos)[:, np.newaxis]
   embeddings = positions * omega(np.arange(dims) // 2, dims)
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
      self.vocab_size = underlying_value(vocab_size, int)
      self.output_dim = underlying_value(output_dim, int)
      self.dropout_rate = underlying_value(dropout_rate, float)
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
      seqlen = tf.shape(inputs)[-1]
      embed_out = self.token_embedding(inputs)

      # embed_out 3d (batch_size, seqlen, output_dim)
      embed_out += self.positional_embedding[:seqlen]

      # positional_embedding 2d (seqlen, output_dim) compressed to (seqlen, output_dim)
      # then broadcasted to (batch_size, seqlen, output_dim), summed with embed_out
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
