import os
import sys

from tensorflow import keras

from bard.midi.raw import Preprocessor
from bard.midi.tokenizer import MidiTokenizer
from bard.midi.dataset import build_dataset

MAX_VELOCITY = 128
MIN_VELOCITY = 1

PROJECT_ROOT = os.path.abspath('../../')

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

def train(model, seq: list, args=None):
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
   # split seq 20/80, pad according to max_inp_seqlen and max_tar_seqlen
   inp, tar, features = build_dataset()
   model.fit()
   # log

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
