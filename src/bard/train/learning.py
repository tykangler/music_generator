import tensorflow as tf
from tensorflow import keras

class LearningSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dim, warmup_steps=4000):
        self.dim = dim
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        min_val = tf.minimum(1 / tf.sqrt(step), step * tf.pow(self.warmup_steps, -1.5))
        return 1 / tf.sqrt(self.dim) * min_val
