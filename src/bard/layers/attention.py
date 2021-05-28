import tensorflow as tf
from tensorflow import keras
import numpy as np

from .utils import underlying_value

class MultiHeadRelativeAttention(keras.layers.Layer):
   """
   An implementation of the multi head attention mechanism, with relative position representations. 
   The input K, and V matrices have the shape of (batch, seqlen, dim). The input Q matrix has the shape (batch, query_seqlen, dim).
   The output shape will equal the shape of the query matrix. 
   :params:
      heads: the number of heads to project q, k, and v matrices
      max_relative_pos: the max relative position that will be considered,
      key_dim: the dimensions of the weighted query and key matrices
      value_dim: the dimensions of the weighted value matrices
      kernel_constraint: weight constraints applied to Q, K, V weights
   """
   def __init__(self, heads, max_relative_pos, key_dim=None, value_dim=None, kernel_constraint=None, **kwargs):
      super().__init__(**kwargs)
      # query and key will have the same dimensions. value may or may not have the same dimensions. if value 
      # not specified, then value = key
      # query may have different seqlen
      self.heads = underlying_value(heads, int) 
      self.key_dim = underlying_value(key_dim, int)
      self.value_dim = underlying_value(value_dim, int)
      self.max_relative_pos = underlying_value(max_relative_pos, int)
      self.kernel_constraint = keras.constraints.get(kernel_constraint)

   def build(self, input_shape):
      dim_input = input_shape[-1]

      # dims calculation
      qk_dim = self.key_dim or dim_input
      v_dim = self.value_dim or dim_input
      assert qk_dim % self.heads == 0, """q, k dims must be a multiple of heads"""
      assert v_dim % self.heads == 0, """v dims must be a multiple of heads"""
      self.head_qk_dim = qk_dim // self.heads
      self.head_v_dim = v_dim // self.heads

      # relative positional encoding
      num_rprs = self.max_relative_pos * 2 + 1
      self.rpr_key_embedding = keras.layers.Embedding(
         num_rprs, self.head_qk_dim, embeddings_constraint=self.kernel_constraint, name="relative embedding (key)")
      self.rpr_value_embedding = keras.layers.Embedding(
         num_rprs, self.head_v_dim, embeddings_constraint=self.kernel_constraint, name="relative embedding (value)")

      # project to heads after applying weights/dense
      self.weights_q = keras.layers.Dense(
         qk_dim, use_bias=False, kernel_constraint=self.kernel_constraint, name="query weights")
      self.weights_k = keras.layers.Dense(
         qk_dim, use_bias=False, kernel_constraint=self.kernel_constraint, name="key weights")
      self.weights_v = keras.layers.Dense(
         v_dim, use_bias=False, kernel_constraint=self.kernel_constraint, name="value weights")

      # concatenated heads passed as input
      self.concat_head_weights = keras.layers.Dense(
         input_shape[-1], kernel_constraint=self.kernel_constraint, name="concat weights")

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
         mask: tensor with shape equal to or broadcastable to (batch, q_seqlen, seqlen) to be applied to each attn head
      :returns:
         attn_scores: Attention scores with shape (batch, q_seqlen, dim), i.e., the attention scores
               for each head in each batch
         attn_weights: Attention weights with shape (batch, heads, q_seqlen, seqlen) 
               applied to the value tensor for each head
      """
      if key is not None:
         key = value
      batch_size, query_seqlen, dim_input = tf.shape(query)
      rpr_lookup = self._generate_rpr_lookup(query_seqlen, self.max_relative_pos)

      # forward pass through weights and split (after for efficiency)
      query = self._project_to_heads(self.weights_q(query))
      key = self._project_to_heads(self.weights_k(key))
      value = self._project_to_heads(self.weights_v(value))   

      # compute attention scores and grab weights
      attn_scores, attn_weights = self._compute_attn(query, value, key, rpr_lookup, mask)

      # transpose and reshape to concat scores for each head in each batch
      attn_scores = tf.transpose(attn_scores, perm=[0, 2, 1, 3]) # (batch_size, q_seqlen, heads, head_dim)
      concat_attn = tf.reshape(attn_scores, (batch_size, -1, dim_input)) # (batch_size, q_seqlen, dim)

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

   def _compute_attn(self, query, value, key, rpr_lookup, mask):
      """

      :params:
         query: tensors of shape (batch, heads, q_seqlen, head_qk_dim)
         value: tensors of shape (batch, heads, seqlen, head_v_dim)
         key: tensors of shape (batch, heads, seqlen, head_qk_dim)
         mask: tensor with shape equal to or broadcastable to (batch, q_seqlen, seqlen) 
               applied after first matmul and just prior to softmax
      :returns:
         attn_scores: Attention scores with shape (..., seqlen, dim), i.e., the attention scores
               for each head in each batch
         attn_weights: Attention weights applied to the value tensor
      """
      # technically, value may have a different dim, resulting in an attention shape of (..., seqlen, dim_v)
      # query and key must have the same dimensions
      key_dims = tf.shape(key)[-1]

      alpha = tf.matmul(query, key, transpose_b=True) 
      rpr_key_embedding = self.rpr_key_embedding(rpr_lookup)
      alpha += self._compute_relative(query, rpr_key_embedding, transpose_embeddings=True)
      alpha /= tf.sqrt(tf.cast(key_dims, tf.float32))

      if mask:
         alpha += mask[:, tf.newaxis, :, :] * -np.inf
      attn_weights = tf.nn.softmax(alpha) # default last axis (key_dims)
      attn_scores = tf.matmul(attn_weights, value)
      rpr_value_embedding = self.rpr_value_embedding(rpr_lookup)
      attn_scores += self._compute_relative(attn_weights, rpr_value_embedding, transpose_embeddings=False)

      return attn_scores, attn_weights

   @classmethod
   def from_config(cls, config):
      return cls(**config)

   def get_config(self):
      config = super().get_config()
      config.update(dict(
         heads=self.heads,
         max_relative_pos=self.max_relative_pos,
         key_dim=self.key_dim,
         value_dim=self.value_dim,
         kernel_constraint=keras.constraints.serialize(self.kernel_constraint)
      ))
      return config
