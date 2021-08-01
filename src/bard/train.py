import os
import sys
import json

from tensorflow import keras

from bard import etl
from bard import constants

from bard.model import SingleVocabTransformer
from bard.layers import learning, loss, metric

def train(model: keras.Model, train_ds, val_ds, train_config):
   callbacks = [
      keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0), # in epochs
      keras.callbacks.TensorBoard(
         log_dir=train_config['log_path'], write_graph=True, histogram_freq=50, # in epochs
         update_freq="epoch", profile_batch=2, embeddings_freq=50), # in epochs
      keras.callbacks.ModelCheckpoint(
         filepath=f"{train_config['checkpoint_path']}/ep{{epoch:02d}}.hdf5", verbose=0,
         save_best_only=False, monitor="val_accuracy", mode="auto", save_freq="epoch")
         # might want to change save freq to batches
   ]
   return model.fit(train_ds,
                    callbacks=callbacks,
                    validation_data=val_ds,
                    epochs=train_config['epochs'])

def create_model(model_config, vocab_size):
   """
   creates the transfomer with given hyperparameters. expects config to have `model` key
   with `initial_params` sub-key, each containing parameters for training objects
   """
   model_config = model_config["initial_params"]
   model = SingleVocabTransformer(vocab_size=vocab_size, **model_config["transformer"])

   lr_sched = learning.LearningSchedule(**model_config["learning_schedule"])
   optimizer = keras.optimizers.Adam(learning_rate=lr_sched, **model_config["optimizer"])
   loss_func = loss.PaddedSparseCategoricalCrossentropy(**model_config["loss"])
   metrics = metric.PaddedSparseTopKCategoricalAccuracy(**model_config["metric"])
   model.compile(
      optimizer=optimizer,
      loss=loss_func,
      metrics=metrics)
   return model

def run(args=None):
   # load config
   with open(constants.config_path) as config_file:
      config = json.load(config_file)
   train_config = config["train"]
   model_config = config["model"]
   midi_path = os.path.join(constants.project_root, train_config['midi_path'])

   # load dataset, and create model
   train_ds, val_ds, _, midi_tokenizer = etl.load(midi_path,
      batch_size=train_config["batch_size"],
      train_size=train_config["train_size"],
      val_size=train_config["val_size"],
      test_size=train_config["test_size"],
      inp_len=train_config["inp_len"],
      tar_len=train_config["tar_len"])
   model = create_model(model_config, midi_tokenizer.vocab_size())

   # fit model
   history = train(model, train_ds, val_ds, train_config)

   # save model
   model_path = os.path.join(constants.project_root, model_config['save_path'], model_config['name'])
   model.save(model_path)

   # save vocabulary
   vocab_path = os.path.join(constants.project_root, model_config['vocab_path'])
   midi_tokenizer.save(vocab_path)

if __name__ == '__main__':
   run(sys.argv[1:])

# TODOs
#
# * [x] Add l2 MaxNorm regularization to layers
#     * ~~Use self.losses~~, weights automatically adjusted as updated
#     * [x] Implement for multihead relative attention
# * [x] Configs for kernel constraints
# * [ ] Clean and Checkup
# * [x] Document all classes and methods with required shapes
# * [ ] Azure Subscription
