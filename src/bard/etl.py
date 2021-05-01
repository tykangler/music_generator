import os
import glob
import mido
import tensorflow as tf
import numpy as np

from .midi import transform

RESOLUTION = 16
MIN_VELOCITY = 1
MAX_VELOCITY = 128
NUM_BINS = 32

TRANSFORMS = [
   transform.VelocityBinner(min_val=MIN_VELOCITY, max_val=MAX_VELOCITY, num_bins=NUM_BINS),
   transform.TimeQuantizer(resolution=RESOLUTION),
   transform.VocabEncoder(resolution=RESOLUTION)
]

def _make_callable(transforms):
   """
   Given a list of callable transformations in `transforms`, returns a callable that
   applies each transformation successively to a given input. All callables must be single 
   argument functions.
   """
   def _callable_delegate(inputs):
      for func in transforms:
         inputs = func(inputs)
      return inputs
   return _callable_delegate

def _get_inp_tar_seq(filename, inp_split, inp_maxlen, tar_maxlen, transforms):
   midi_sequence = mido.MidiFile(filename)
   track = transforms(midi_sequence)

   inp_len = int(inp_split * len(track))
   tar_len = len(track) - inp_len

   inp_seq, tar_seq = track[:inp_len], track[inp_len:]
   inp_seq = np.pad(inp_seq, (0, inp_maxlen - inp_len))
   tar_seq = np.pad(tar_seq, (0, tar_maxlen - tar_len))
   return inp_seq, tar_seq

def _get_split_midi_data(midi_filenames, inp_split, max_seqlen, transforms):
   inputs, targets = [], []
   inp_maxlen = inp_split * max_seqlen
   tar_maxlen = max_seqlen - inp_maxlen
   transforms = _make_callable(transforms)
   for filename in midi_filenames:
      inp_seq, tar_seq = _get_inp_tar_seq(filename, inp_split, inp_maxlen, tar_maxlen, transforms)
      inputs.append(inp_seq)
      targets.append(tar_seq)
   return inputs, targets

def _get_dataset_splits(train_size, val_size, test_size):
   dataset_splits: list[float] = [train_size, val_size, test_size]
   assert dataset_splits.count(None) <= 1
   if dataset_splits.count(None) == 0: 
      assert np.sum(dataset_splits) == 1
   else:
      split_sums = np.sum([split for split in dataset_splits if split is not None])
      idxNone = dataset_splits.index(None)
      dataset_splits[idxNone] = 1.0 - split_sums
   return dataset_splits

def load(midi_path, *, train_size=None, val_size=None, test_size=None, inp_split, max_seqlen):
   """
   returns a 3-tuple of `tf.Dataset` each returning `(input_seq, target_seq)`, representing train, 
   validation, and test portions of the overall dataset. `input_seq` represents the `inp_split` 
   portion of each midi sequence in `midi_path`.

   collects all midi files in `midi_path` recursively, and applies 
   transformations. The size of `input_seq` for each dataset is `inp_split * len(midi sequence)`, 
   and is padded to `inp_split * max_seqlen`.

   params:
      midi_path: path to directory containing midi files
      train_size: float representing portion of overall dataset allocated to training
      val_size: float representing portion allocated to validation
      test_size: float representing portion allocated to test
      inp_split: portion of each midi sequence allocated to the input sequence, the rest is allocated
         to the target sequence
      max_seqlen: maximum sequence length of read midi sequence. only messages up to `max_seqlen` 
         will be considered in input-target split. `inp_split * max_seqlen` is the maximum size of
         the input sequence, and `1 - (inp_split * max_seqlen)` is the maximum size of the target 
         sequence.
   """
   # get midi files
   midi_filenames = np.shuffle(glob.glob(f'{midi_path}/**/*.midi', recursive=True))

   # get train, validation, and test sizes
   train_size, val_size, _ = _get_dataset_splits(train_size, val_size, test_size)
   train_size = int(train_size * len(midi_filenames))
   val_size = int(val_size * len(midi_filenames))
   
   # get input and target sequences
   inputs, targets = _get_split_midi_data(midi_filenames, inp_split, max_seqlen, TRANSFORMS)

   # construct datasets
   full_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
   train_dataset = full_dataset.take(train_size)
   val_dataset = full_dataset.skip(train_size)
   test_dataset = val_dataset.skip(val_size)
   val_dataset = val_dataset.take(val_size)
   return train_dataset, val_dataset, test_dataset
