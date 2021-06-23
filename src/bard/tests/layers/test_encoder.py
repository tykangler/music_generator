import tensorflow as tf
from tensorflow import keras
import numpy as np

from bard.layers import encoder

NUM_HEADS = 8

def create_encoder():
   return encoder.Encoder(
      heads=NUM_HEADS,
      max_relative_pos=20,
      ffnn_dim=50,
      dropout_rate=0.2,
      kernel_constraint=keras.constraints.MaxNorm()
   )

def create_stack():
   return encoder.EncoderStack(
      units=3,
      heads=NUM_HEADS,
      ffnn_dim=50,
      max_relative_pos=20,
      dropout_rate=0.2,
      kernel_constraint=keras.constraints.MaxNorm()
   )

def test_forward_pass_creates_correct_shape():
   layer = create_encoder()
   # (batch = 2, seqlen = 4, qk_dim = 16)
   inputs = tf.random.uniform([2, 4, 16])
   out, attn_weights = layer(inputs, None, False)

def test_encoder_weights_equal_mask():
   layer = create_encoder()
   inputs = tf.random.uniform([2, 4, 16])
   padding_mask = tf.constant(
      [[0, 0, 0, 1],
       [1, 0, 0, 1],
       [1, 0, 1, 0],
       [1, 1, 0, 0]]
   )[tf.newaxis, ...]
   out, attn_weights = layer(inputs, padding_mask, False)
   zero_padding_weights = tf.cast(tf.equal(attn_weights, 0), tf.int32)
   assert bool(tf.reduce_all(tf.equal(zero_padding_weights, padding_mask)))

def test_encoder_config_is_reusable():
   layer = create_encoder()
   config = layer.get_config()
   new_layer = encoder.Encoder.from_config(config)
   assert config == new_layer.get_config()

def test_encoder_stack_creates_correct_shape():
   stack = create_stack()
   inputs = tf.random.uniform([2, 4, 16])
   out, attn_weights = stack(inputs, None, False)
   assert all(tf.shape(out) == tf.shape(inputs))
   for weights in attn_weights.values():
      assert all(tf.shape(weights) == [2, NUM_HEADS, 4, 4])

def test_encoder_stack_config_is_reusable():
   stack = create_stack()
   config = stack.get_config()
   new_stack = encoder.EncoderStack.from_config(config)
   assert config == new_stack.get_config()
