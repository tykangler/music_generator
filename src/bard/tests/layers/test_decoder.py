import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from bard.layers import decoder

NUM_HEADS = 8

def create_decoder():
   return decoder.Decoder(
      heads=NUM_HEADS,
      max_relative_pos=20,
      ffnn_dim=50,
      dropout_rate=0.2,
      kernel_constraint=keras.constraints.MaxNorm()
   )

def create_stack():
   return decoder.DecoderStack(
      units=3,
      heads=NUM_HEADS,
      ffnn_dim=50,
      max_relative_pos=20,
      dropout_rate=0.2,
      kernel_constraint=keras.constraints.MaxNorm()
   )

def test_forward_pass_creates_correct_shape():
   layer = create_decoder()
   # (batch=2, q_seqlen=4, qk_dim=16)
   inputs = tf.random.uniform([2, 4, 16])
   enc_kv = tf.random.uniform([2, 9, 16])
   out, self_attn_weights, cross_attn_weights = layer(inputs, enc_kv, None, None, True)
   assert all(tf.shape(out) == tf.shape(inputs))
   assert all(tf.shape(self_attn_weights) == [2, NUM_HEADS, 4, 4])
   assert all(tf.shape(cross_attn_weights) == [2, NUM_HEADS, 4, 9])

def test_decoder_weights_equal_mask():
   layer = create_decoder()
   # (batch=2, q_seqlen=4, qk_dim=16)
   query = tf.random.uniform([2, 4, 16])
   enc_kv = tf.random.uniform([2, 6, 16])
   lookahead_mask = tf.constant(
      [[1, 0, 1, 0],
       [1, 1, 0, 0],
       [0, 0, 0, 1],
       [1, 1, 1, 0]]
   )[tf.newaxis, ...]
   padding_mask = tf.constant(
      [[0, 0, 0, 1, 1, 1],
       [1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 1],
       [1, 1, 0, 0, 1, 1]]
   )[tf.newaxis, ...]
   out, self_attn_weights, cross_attn_weights = layer(
      query, enc_kv, padding_mask, lookahead_mask, True)
   zero_lookahead_weights = tf.cast(tf.equal(self_attn_weights, 0), tf.int32)
   assert bool(tf.reduce_all(tf.equal(zero_lookahead_weights, lookahead_mask)))
   zero_padding_weights = tf.cast(tf.equal(cross_attn_weights, 0), tf.int32)
   assert bool(tf.reduce_all(tf.equal(zero_padding_weights, padding_mask)))

def test_decoder_stack_creates_correct_shape():
   stack = create_stack()
   inputs = tf.random.uniform([2, 4, 16])
   enc_kv = tf.random.uniform([2, 9, 16])
   out, attn_weights = stack(inputs, enc_kv, None, None, True)
   # attn_weights = { 'decoder_{i}': (self_weights, cross_weights) }
   assert all(tf.shape(out) == tf.shape(inputs))
   for self_weights, cross_weights in attn_weights.values():
      assert all(tf.shape(self_weights) == [2, NUM_HEADS, 4, 4])
      assert all(tf.shape(cross_weights) == [2, NUM_HEADS, 4, 9])
