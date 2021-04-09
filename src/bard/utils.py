import json
import os

from tensorflow import keras

from .model import MusicTransformer
from .layers import learning, loss, metric

def load_config(config_dir='../../'):
   config_dir = os.path.abspath(config_dir)
   with open(rf'{config_dir}/config.json') as config_file:
      config_json = json.load(config_file)
   return config_json

def calculate_vocab_size(config):
   """
   expects a midi config
   """
   instruments = len(config["instruments"])
   notes = config["notes"]
   velocities = config["velocity_bins"]
   num_rests = config["rest_resolution"]
   # extra (+notes) for note offs
   return instruments * notes * velocities + num_rests + notes

def create_model(config):
   """
   creates the transfomer with given hyperparameters. expects a `model` config with `midi` and 
   `initial_params` keys
   """
   midi_config = config['midi']
   model_config = config['initial_params']
   transformer = MusicTransformer(
      vocab_size=calculate_vocab_size(midi_config), **model_config['model'])

   lr_sched = learning.LearningSchedule(**model_config['learning_schedule'])
   optimizer = keras.optimizers.Adam(learning_rate=lr_sched, **model_config['optimizer'])
   loss_func = loss.PaddedSparseCategoricalCrossentropy(**model_config['loss'])
   metrics = metric.PaddedSparseTopKCategoricalAccuracy(**model_config['metric'])
   transformer.compile(
      optimizer=optimizer, 
      loss=loss_func, 
      metrics=metrics)
   return transformer
