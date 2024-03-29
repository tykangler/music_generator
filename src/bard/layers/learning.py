import tensorflow as tf
from tensorflow import keras

from .utils import underlying_value

class LearningSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dim, warmup_steps=4000):
        self.dim = underlying_value(dim, int)
        self.warmup_steps = underlying_value(warmup_steps, int)

    def __call__(self, epoch):
        min_val = tf.minimum(tf.math.rsqrt(epoch), epoch * tf.pow(self.warmup_steps, -1.5))
        return 1 / tf.math.rsqrt(self.dim) * min_val

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            dim=self.dim,
            warmup_steps=self.warmup_steps
        ))
