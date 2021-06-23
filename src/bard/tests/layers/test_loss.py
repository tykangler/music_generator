import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from bard.layers import loss

from .sampling import *

def test_unpadded_loss_equals_padded_loss_with_padding():
   y_true, y_pred = sample_true_pred_padded()
   unpadded_true, unpadded_pred = unpad_true_pred(y_true, y_pred)

   padded_loss = loss.PaddedSparseCategoricalCrossentropy()
   raw_loss = SparseCategoricalCrossentropy()

   padded_res = padded_loss(y_true, y_pred).numpy()
   raw_res = raw_loss(unpadded_true, unpadded_pred).numpy()

   assert np.isclose(padded_res, raw_res)

def test_unpadded_loss_equals_padded_loss_without_padding():
   y_true, y_pred = sample_true_pred_padded()
   unpadded_true, unpadded_pred = unpad_true_pred(y_true, y_pred)

   padded_loss = loss.PaddedSparseCategoricalCrossentropy()
   raw_loss = SparseCategoricalCrossentropy()

   padded_res = padded_loss(unpadded_true, unpadded_pred).numpy()
   raw_res = raw_loss(unpadded_true, unpadded_pred).numpy()
   assert np.isclose(padded_res, raw_res)
