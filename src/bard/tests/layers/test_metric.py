from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import numpy as np

from bard.layers import metric
from .sampling import *

def test_one_pass_generates_correct_metric_with_padding():
   y_true, y_pred = sample_true_pred_padded()
   padded_metric = metric.PaddedSparseTopKCategoricalAccuracy(k=3)
   padded_metric.update_state(y_true, y_pred)

   raw_metric = SparseTopKCategoricalAccuracy(k=3)
   raw_metric.update_state(*unpad_true_pred(y_true, y_pred))

   raw_result = raw_metric.result().numpy()
   padded_result = padded_metric.result().numpy()
   assert np.isclose(raw_result, padded_result)

def test_one_pass_generates_correct_metric_without_padding():
   y_true, y_pred = unpad_true_pred(*sample_true_pred_padded())

   padded_metric = metric.PaddedSparseTopKCategoricalAccuracy(k=3)
   raw_metric = SparseTopKCategoricalAccuracy(k=3)

   padded_metric.update_state(y_true, y_pred)
   raw_metric.update_state(y_true, y_pred)

   raw_result = raw_metric.result().numpy()
   padded_result = padded_metric.result().numpy()
   assert np.isclose(raw_result, padded_result)

def test_multiple_passes_generates_correct_metric_with_padding():
   padded_metric = metric.PaddedSparseTopKCategoricalAccuracy(k=3)
   raw_metric = SparseTopKCategoricalAccuracy(k=3)

   for i in range(10):
      y_true, y_pred = sample_true_pred_padded()
      unpadded_true, unpadded_pred = unpad_true_pred(y_true, y_pred)
      raw_metric.update_state(unpadded_true, unpadded_pred)
      padded_metric.update_state(y_true, y_pred)

   raw_result = raw_metric.result().numpy()
   padded_result = padded_metric.result().numpy()
   assert np.isclose(raw_result, padded_result)

def test_multiple_passes_generates_correct_metric_without_padding():
   padded_metric = metric.PaddedSparseTopKCategoricalAccuracy(k=3)
   raw_metric = SparseTopKCategoricalAccuracy(k=3)

   for i in range(10):
      y_true, y_pred = sample_true_pred_padded()
      unpadded_true, unpadded_pred = unpad_true_pred(y_true, y_pred)
      raw_metric.update_state(unpadded_true, unpadded_pred)
      padded_metric.update_state(unpadded_true, unpadded_pred)

   raw_result = raw_metric.result().numpy()
   padded_result = padded_metric.result().numpy()

def test_config_is_reusable():
   m = metric.PaddedSparseTopKCategoricalAccuracy(k=3)
   new_metric = metric.PaddedSparseTopKCategoricalAccuracy.from_config(m.get_config())
   assert m.get_config() == new_metric.get_config()
