import tensorflow as tf
from tensorflow import keras

from .utils import underlying_value

class PaddedSparseTopKCategoricalAccuracy(keras.metrics.Metric):
    def __init__(self, k=5, name="padded_sparse_top_k_categorical_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = underlying_value(k, int)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        params:
            y_true: input of shape (batch, seqlen)
            y_pred: input of shape (batch, seqlen, dim)
        """
        _, top_k_labels = tf.math.top_k(y_pred, self.k)
        label_matches = tf.reduce_any(tf.equal(y_true[..., tf.newaxis], top_k_labels), axis=-1) # (batch, seqlen)

        not_padded = tf.not_equal(y_true, 0) # [[True, True, True ... False, False], [...]]
        not_padded_matches = tf.cast(tf.logical_and(not_padded, label_matches), tf.float32)

        # implement sample weights using tf.dot
        self.total.assign_add(tf.reduce_sum(not_padded_matches))
        self.count.assign_add(tf.reduce_sum(tf.cast(not_padded, tf.float32)))

    def result(self):
        return self.total / self.count

    def get_config(self):
        config = super().get_config()
        config.update(dict(k=self.k))
        return config
