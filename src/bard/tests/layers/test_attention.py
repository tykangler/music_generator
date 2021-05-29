import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pytest
import tensorflow as tf
from tensorflow import keras
from bard.layers import attention

NUM_HEADS = 8

def create_self_attn():
   "creates attention layer with inferred key and value dimensions"
   return attention.MultiHeadRelativeAttention(
      heads=NUM_HEADS,
      max_relative_pos=20,
      key_dim=None,
      value_dim=None,
      kernel_constraint=keras.constraints.MaxNorm()
   )

def create_cross_attn():
   return attention.MultiHeadRelativeAttention(
      heads=NUM_HEADS,
      max_relative_pos=20,
      key_dim=None,
      value_dim=None,
      kernel_constraint=keras.constraints.MaxNorm(),
      use_relative_embed=False
   )

# TEST METHODS ------------------

def test_self_attention_produces_correct_shape():
   attn = create_self_attn()
   # (batch=2, q_seqlen=4, dim=16)
   inputs = tf.reshape(tf.range(128), (2, 4, 16))
   out, weights = attn(inputs, inputs, inputs)
   assert all(tf.shape(inputs) == tf.shape(out))
   assert all(tf.shape(weights) == [2, NUM_HEADS, 4, 4])

def test_config_is_reusable():
   attn = create_self_attn()
   config = attn.get_config()
   recreated_attn = attention.MultiHeadRelativeAttention.from_config(config)
   assert config == recreated_attn.get_config()

def test_config_is_reusable_with_tensors():
   attn = attention.MultiHeadRelativeAttention(
      heads=tf.constant(8),
      max_relative_pos=tf.constant(20),
      key_dim=None,
      value_dim=None,
      kernel_constraint=keras.constraints.MaxNorm()
   )
   config = attn.get_config()
   recreated_attn = attention.MultiHeadRelativeAttention.from_config(config)
   return config == recreated_attn.get_config()

def test_raises_when_dim_not_multiple_of_heads():
   attn = create_self_attn()
   with pytest.raises(AssertionError):
      inputs = tf.reshape(tf.range(144), (2, 4, 18))
      out, weights = attn(inputs)

def test_cross_attention_produces_correct_shape():
   attn = create_cross_attn()
   # (batch=2, q_seqlen=4, dim=16)
   # (batch=2, kv_seqlen=20, dim=16)
   query = tf.reshape(tf.range(128), (2, 4, 16))
   key_value = tf.reshape(tf.range(640), (2, 20, 16))
   out, weights = attn(query, key_value, key_value)
   assert all(tf.shape(query) == tf.shape(out))
   assert all(tf.shape(weights) == [2, NUM_HEADS, 4, 20])
