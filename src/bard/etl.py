import os
import glob
import mido
import tensorflow as tf
import numpy as np

def get_inp_tar_seq(filename, inp_split, inp_maxlen, tar_maxlen, msg_filter):
   midi_sequence = mido.MidiFile(filename)
   track = msg_filter(midi_sequence.tracks[1])

   inp_len = int(inp_split * len(track))
   tar_len = len(track) - inp_len

   inp_seq, tar_seq = track[:inp_len], track[inp_len:]
   inp_seq = np.pad(inp_seq, (0, inp_maxlen - inp_len))
   tar_seq = np.pad(tar_seq, (0, tar_maxlen - tar_len))
   return inp_seq, tar_seq

def get_split_midi_data(midi_filenames, inp_split, max_seqlen, msg_filter):
   inputs, targets = [], []
   inp_maxlen = inp_split * max_seqlen
   tar_maxlen = max_seqlen - inp_maxlen

   for filename in midi_filenames:
      inp_seq, tar_seq = get_inp_tar_seq(filename, inp_split, inp_maxlen, tar_maxlen, msg_filter)
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

def build_dataset(midi_dir, *, train_size=None, val_size=None, test_size=None, 
                  inp_split, max_seqlen, msg_filter):
   """
   builds dataset and returns a 3-tuple of `tf.Dataset` each returning `(input_seq, target_seq)`,
   representing train, validation, and test portions of the overall dataset.
   """
   train_size, val_size, test_size = _get_dataset_splits(train_size, val_size, test_size)

   midi_filenames = np.shuffle(glob.glob(f'{midi_dir}/[0-9]*/*'))
   train_size = int(train_size * len(midi_filenames))
   val_size = int(val_size * len(midi_filenames))
   
   inputs, targets = get_split_midi_data(midi_filenames, inp_split, max_seqlen, msg_filter)
   full_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
   train_dataset = full_dataset.take(train_size)
   val_dataset = full_dataset.skip(train_size)
   test_dataset = val_dataset.skip(val_size)
   val_dataset = val_dataset.take(val_size)

   return train_dataset, val_dataset, test_dataset
