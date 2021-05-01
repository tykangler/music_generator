import os
import sys
import json

from tensorflow import keras

from bard.midi.tokenizer import MidiTokenizer
from bard.etl import build_dataset

MAX_VELOCITY = 128
MIN_VELOCITY = 1

PROJECT_ROOT = os.path.abspath('../..')


def train(log_dir, checkpoint_dir):
   callbacks = [
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0), # in epochs
      keras.callbacks.TensorBoard(
         log_dir=log_dir, write_graph=True, histogram_freq=50, # in epochs
         update_freq='epoch', profile_batch=2, embeddings_freq=50), # in epochs
      keras.callbacks.ModelCheckpoint(
         filepath=f'{checkpoint_dir}/ep{{epoch:02d}}.hdf5', verbose=0, 
         save_best_only=False, monitor='val_accuracy', mode='auto', save_freq='epoch') 
         # might want to change save freq to batches
   ]

if __name__ == '__main__':
   # load config
   config_path = f'{PROJECT_ROOT}/config.json'
   with open(config_path) as config_file:
      config = json.load(config_file)
   log_dir = f'{PROJECT_ROOT}/{config["app"]["log_dir"]}'
   checkpoint_dir = f'{PROJECT_ROOT}/{config["app"]["checkpoint_dir"]}'
   
   # create model
   train(sys.argv)
   
# TODOs
# 
# * [x] Add l2 MaxNorm regularization to layers
#     * ~~Use self.losses~~, weights automatically adjusted as updated
#     * [x] Implement for multihead relative attention
# * [x] Configs for kernel constraints 
# * [ ] ~~Pass features into model~~ Later
# * [ ] Clean and Checkup
# * [ ] ~~Modify `test_step()` to use predict method using seqlen of validation sequence~~
# * [x] Document all classes and methods with required shapes
