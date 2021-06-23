import pytest
import tensorflow as tf
import numpy as np

from bard.layers import masking

@pytest.mark.parametrize("input_shape", [
   [5], [10], [2, 4], [2, 4, 6], [8, 10, 2, 3]
])
def test_padding_mask_has_correct_shape(input_shape):
   inputs = np.random.randint(low=0, high=10, size=input_shape)
   mask = masking.create_padding_mask(inputs)
   assert all(tf.shape(mask) == tf.shape(inputs))

def test_padding_mask_placed_values_correctly():
   inputs = tf.constant([
      [1, 2, 3, 0, 0],
      [1, 3, 0, 0, 0],
      [1, 2, 3, 4, 0]
   ])
   correct_mask = tf.constant([
      [0, 0, 0, 1, 1],
      [0, 0, 1, 1, 1],
      [0, 0, 0, 0, 1]
   ], dtype=tf.float32)
   mask = masking.create_padding_mask(inputs)
   assert tf.reduce_all(mask == correct_mask)

@pytest.mark.parametrize("input_shape", [
   5, 10, 2, 4, 2, 4, 6, 8, 10, 2, 3
])
def test_lookahead_mask_has_correct_shape(input_shape):
   mask = masking.create_lookahead_mask(input_shape)
   assert all(dim == input_shape for dim in tf.shape(mask))

@pytest.mark.parametrize("input_shape", [
   5, 10, 2, 4, 2, 4, 6, 8, 10, 2, 3
])
def test_lookahead_mask_placed_values_correctly(input_shape):
   mask = masking.create_lookahead_mask(input_shape)
   for i in range(len(mask)):
      for j in range(len(mask)):
         if j - i > 0:
            assert mask[i, j] == 1
         else:
            assert mask[i, j] == 0
