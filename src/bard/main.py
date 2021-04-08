import collections
import os
import sys

import tensorflow as tf
from tensorflow import keras

from bard.layers.masking import create_decoder_mask, create_padding_mask
from bard.midi.raw import Preprocessor
from bard.midi.tokenizer import MidiTokenizer
from bard.midi.dataset import build_dataset

MAX_VELOCITY = 128
MIN_VELOCITY = 1

PROJECT_ROOT = os.path.abspath('../../')

def dispatch(model, seq: list, train: bool, args=None):
   """
   dispatches to training and prediction, after preprocessing

   params:
      `seq`: tf.tensor or np.ndarray with shape (batch, seqlen)
   """
   args = [] if args is None else args
   preprocessor = Preprocessor(steps=[])
   tokenizer = MidiTokenizer()
   seq = preprocessor(seq)
   seq = tokenizer.encode(seq) # here, inp will have start, but not end, and tar will have end, not start
   if train:
      # split seq 20/80, pad according to max_inp_seqlen and max_tar_seqlen
      inp, tar, features = build_dataset()
      model.fit()
      # log
   else:
      predict(model, seq, tokenizer.end_token(), tokenizer.start_token())
      # log

# train -------------

# change log_dir and checkpoint filepath at future point
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0), # in epochs
    keras.callbacks.TensorBoard(
        log_dir='logs', write_graph=True, histogram_freq=50, # in epochs
        update_freq='epoch', profile_batch=2, embeddings_freq=50), # in epochs
    keras.callbacks.ModelCheckpoint(
        filepath='model_checkpoints/ep{epoch:02d}-val_acc{val_accuracy:.2f}.hdf5', verbose=0, 
        save_best_only=False, monitor='val_accuracy', mode='auto', save_freq='epoch') # might want to change to batches
]

# predict --------------

def predict(model, input_seq: list, end_token: int, start_token: int, max_len: int):
   """
   infers a continuation of the input sequence until the given end token is reached, or until the continuation
   has a sequence length of max_len.

   params:
      input_seq: a list with shape (seqlen), with elements representing the preprocessed notes/waits of a midi sequence
      end_token: the designated token to end the sequence with
      start_token: the designated start token
      max_len: maximum length of the continued sequence
   """
   output_seq = tf.constant([start_token])[tf.newaxis, :] # (batch=1, q_seqlen=1)
   input_padding_mask = create_padding_mask(input_seq)[:, tf.newaxis, :] # (batch=1, q_seqlen=1 (broadcast), seqlen)
   AttentionWeights = collections.namedtuple('AttentionWeights', 
      ['enc_weights', 'dec_weights', 'encdec_weights'])
   while tf.size(output_seq) != max_len and output_seq[-1, -1] != end_token:
      decoder_mask = create_decoder_mask(output_seq) # (batch=1, seqlen, seqlen)
      # generated_seq is made up of ignored first n - 1 tokens, with the final nth token being the prediction
      generated_seq, *weights = model(
         inputs=input_seq, 
         targets=output_seq,
         # features=??,
         padding_mask=input_padding_mask,
         lookahead_mask=decoder_mask,
         training=False) # (batch=1, q_seqlen, vocab_size)
      last_notes = generated_seq[:, -1:, :] # (batch=1, 1, vocab_size)
      generated_labels = tf.argmax(last_notes, axis=-1) 
      # (batch=1, 1), what if gen label = 0, model should be trained to not gen 0
      output_seq = tf.concat([output_seq, generated_labels], axis=-1)
   return output_seq, AttentionWeights(*weights)


if __name__ == '__main__':
   args = sys.argv

# TODOs
# 
# * [x] Add l2 MaxNorm regularization to layers
#     * ~~Use self.losses~~, weights automatically adjusted as updated
#     * [x] Implement for multihead relative attention
# * [x] Configs for kernel constraints 
# * [ ] Pass features into model
# * [ ] Clean and Checkup
# * [ ] Modify `test_step()` to use predict method using seqlen of validation sequence
# * [ ] Document all classes and methods with required shapes
