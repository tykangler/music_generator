import tensorflow as tf
from tensorflow import keras

from .utils import underlying_value

class PaddedSparseCategoricalCrossentropy(keras.losses.SparseCategoricalCrossentropy):
    def __init__(
        self,
        name="padded_sparse_categorical_cross_entropy",
        **kwargs):
        super().__init__(name=name, **kwargs)

    def _mask_metric_where_padded(self, raw_metric, y_true):
        """
        finds and masks sequence pads in y_true, and applies the masked result to raw_metric
        """
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float64)
        loss_tensor_masked = mask * raw_metric
        return tf.reduce_sum(loss_tensor_masked) / tf.reduce_sum(mask)

    def call(self, y_true, y_pred):
        # y_true = [[1, 2, 3, <end>, 0, 0], [5, 2, 5, 2, 4, <end>]]
        # y_pred =
        #   [[[...], [...], [...], [...], no loss calc -> [...], [...]],
        #    [[...], [...], [...], [...], [...], [...]]]
        # get loss using avg reduction, mask out padded portions, recalculate average

        loss_val = super().call(y_true, y_pred)
        return self._mask_metric_where_padded(loss_val, y_true)

    def get_config(self):
        config = super().get_config()
        return config
