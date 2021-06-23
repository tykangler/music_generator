import tensorflow as tf
import numpy as np

BATCH = 2
SEQLEN = 16
DIM = 8
NUM_PAD = 4

def unpad_true_pred(true, pred):
   not_zero = tf.not_equal(true, 0)[0]
   true = tf.boolean_mask(true, not_zero, axis=1)
   pred = tf.boolean_mask(pred, not_zero, axis=1)
   return true, pred

def sample_true_pred_padded():
   y_true = np.random.randint(low=1, high=DIM, size=[BATCH, SEQLEN - NUM_PAD])
   y_true = tf.constant(np.concatenate([y_true, np.zeros([BATCH, NUM_PAD])], axis=-1), dtype=tf.int32)
   y_pred = tf.constant(np.random.random_sample([BATCH, SEQLEN, DIM]))
   return y_true, y_pred
