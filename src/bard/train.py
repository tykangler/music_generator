import os
import sys
import json

from tensorflow import keras

from bard import etl

from bard.model import MusicTransformer
from bard.layers import learning, loss, metric


MAX_VELOCITY = 128
MIN_VELOCITY = 1

PROJECT_ROOT = os.path.abspath("../..")

def train(model: keras.Model, *, train_ds, val_ds, log_dir, checkpoint_dir, **kwargs):
   callbacks = [
      keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0), # in epochs
      keras.callbacks.TensorBoard(
         log_dir=log_dir, write_graph=True, histogram_freq=50, # in epochs
         update_freq="epoch", profile_batch=2, embeddings_freq=50), # in epochs
      keras.callbacks.ModelCheckpoint(
         filepath=f"{checkpoint_dir}/ep{{epoch:02d}}.hdf5", verbose=0, 
         save_best_only=False, monitor="val_accuracy", mode="auto", save_freq="epoch") 
         # might want to change save freq to batches
   ]
   model.fit(train_ds, callbacks=callbacks, validation_data=val_ds)

def create_model(config, vocab_size):
   """
   creates the transfomer with given hyperparameters. expects a `model` config with `vocabulary` and 
   `model` keys, the `model` key containing an `initial_params` sub-key

   ```
   {
      ...
      "vocabulary": { ... },
      "model: {
         "initial_params": { 
            "layer_1": { ... },
            "layer_2": { ... },
            ...
         },
      },
      ...
   }
   ```
   """
   model_config = config["model"]["initial_params"]
   model = MusicTransformer(
      vocab_size=vocab_size, **model_config["model"])

   lr_sched = learning.LearningSchedule(**model_config["learning_schedule"])
   optimizer = keras.optimizers.Adam(learning_rate=lr_sched, **model_config["optimizer"])
   loss_func = loss.PaddedSparseCategoricalCrossentropy(**model_config["loss"])
   metrics = metric.PaddedSparseTopKCategoricalAccuracy(**model_config["metric"])
   model.compile(
      optimizer=optimizer, 
      loss=loss_func, 
      metrics=metrics)
   return model

def main(args):
   # load config
   config_path = f"{PROJECT_ROOT}/config.json"
   with open(config_path) as config_file:
      config = json.load(config_file)
   train_config = config["app"]
   log_dir = f"{PROJECT_ROOT}/{train_config['log_dir']}"
   checkpoint_dir = f"{PROJECT_ROOT}/{train_config['checkpoint_dir']}"
   midi_path = f"{PROJECT_ROOT}/{train_config['midi_dir']}"

   train_ds, val_ds, test_ds, midi_tokenizer = etl.load(midi_path, 
      batch_size=train_config["batch_size"],
      train_size=train_config["train_size"],
      val_size=train_config["val_size"],
      test_size=train_config["test_size"],
      inp_len=train_config["inp_len"],
      tar_len=train_config["tar_len"])
   model = create_model(config, midi_tokenizer.vocab_size())
   
   # create model
   train(model, 
      train_ds=train_ds, 
      val_ds=val_ds, 
      log_dir=log_dir, 
      checkpoint_dir=checkpoint_dir)

if __name__ == '__main__':
   main(sys.argv)
   
# TODOs
# 
# * [x] Add l2 MaxNorm regularization to layers
#     * ~~Use self.losses~~, weights automatically adjusted as updated
#     * [x] Implement for multihead relative attention
# * [x] Configs for kernel constraints 
# * [x] ~~Pass features into model~~ Later
# * [ ] Clean and Checkup
# * [x] ~~Modify `test_step()` to use predict method using seqlen of validation sequence~~
# * [x] Document all classes and methods with required shapes
# * [ ] Get AWS EC2 instance https://aws.amazon.com/getting-started/hands-on/train-deep-learning-model-aws-ec2-containers/
