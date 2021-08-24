import tensorflow as tf
import numpy as np
import mido
from bard import etl

CONFIG = {
   'inp_len': 1000,
   'tar_len': 50000,
   'rest_resolution': 32,
   'velocity_bins': 32,
   'batch_size': 32,
   'train_size': 0.8,
   'test_size': 0.1
}

def test_load():
   ds = etl.load('../../../data/maestro-v3.0.0')
