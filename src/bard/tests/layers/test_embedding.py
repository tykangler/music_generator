import numpy as np
from bard.layers import embedding
import tensorflow as tf

def test_positional_encoding_values():
   values = embedding.positional_embeddings(max_pos=100, dims=32)
   for pos in range(len(values)):
      for i in range(len(values[pos])):
         if i % 2 == 0:
            assert np.isclose(values[pos, i], np.sin(pos / np.power(10000, 2 * (i // 2) / 32)))
         else:
            assert np.isclose(values[pos, i], np.cos(pos / np.power(10000, 2 * (i // 2) / 32)))

def test_layer_forward_pass_creates_correct_shape():
   vocab_size, output_dim = 4, 16
   layer = embedding.Embedding(vocab_size=vocab_size, output_dim=output_dim)
   inputs = tf.random.shuffle(tf.repeat(tf.range(vocab_size), vocab_size))
   inputs = tf.reshape(inputs, [2, -1])
   out = layer(inputs, training=True)
   desired_shape = tf.concat([tf.shape(inputs), [output_dim]], axis=-1)
   assert all(tf.shape(out) == desired_shape)

def test_embedding_config_is_reusable():
   vocab_size, output_dim = 4, 16
   layer = embedding.Embedding(vocab_size=vocab_size, output_dim=output_dim)
   config = layer.get_config()
   new_layer = embedding.Embedding.from_config(config)
   return new_layer.get_config() == config
