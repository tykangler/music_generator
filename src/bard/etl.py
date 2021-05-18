# input pipeline steps:
# 1. dataset of file names
# 2. map -> mido.MidiFile(filename), decode into vocab (rest_b:{}, note_: ... ), tf.py_function eager
# 3. map -> split into inp, tar (this can be separate to allow for vectorization, straight split)

import glob
import mido
import tensorflow as tf
import numpy as np

from .midi import transform, tokenizer

RESOLUTION = 16
MIN_VELOCITY = 1
MAX_VELOCITY = 128
NUM_VELOCITY_BINS = 32

PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2

PREFETCH_BUFFER_SIZE = 2 # at least 1 for the next train step (1 batch processed per train step)

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

transforms = _make_callable([
   transform.VelocityBinner(min_val=MIN_VELOCITY, max_val=MAX_VELOCITY, num_bins=NUM_VELOCITY_BINS),
   transform.TimeQuantizer(resolution=RESOLUTION),
   transform.Stringify(resolution=RESOLUTION)
])

def _get_dataset_splits(train_size: float, val_size: float, test_size: float):
   dataset_splits: list[float] = [train_size, val_size, test_size]
   assert dataset_splits.count(None) <= 1
   if dataset_splits.count(None) == 0: 
      assert np.sum(dataset_splits) == 1
   else:
      split_sums = np.sum([split for split in dataset_splits if split is not None])
      idxNone = dataset_splits.index(None)
      dataset_splits[idxNone] = 1.0 - split_sums
   return dataset_splits

def _process_raw_midi(filename: tf.Tensor):
   """
   processes midi filename at `filename` and returns a tensor with string representations of the
   midi sequence.
   """
   # run eagerly
   processed = mido.MidiFile(filename.numpy())
   return tf.constant(transforms(processed), dtype=tf.string)

def _tokenize_and_split(midi_tokenizer: tokenizer.MidiTokenizer, seq: tf.Tensor, 
                        inp_len: int, tar_len: int):
   """
   will tokenize, split to input and target, prepend and append '<start>' and '<end>' tokens 
   respectively and pad resulting sequence according to the inp_len and tar_len
   """
   tokenized = midi_tokenizer.encode(seq)
   inp, tar = tokenized[:inp_len], tokenized[inp_len:(inp_len + tar_len)]
   inp = tf.concat([START_TOKEN], inp, [END_TOKEN], axis=-1)
   tar = tf.concat([START_TOKEN], tar, [END_TOKEN], axis=-1)
   return inp, tar

def _optimize_dataset(ds: tf.data.Dataset):
   return ds.cache().prefetch(buffer_size=PREFETCH_BUFFER_SIZE)

def _create_dataset(filenames: list, batch_size: int, inp_len: int, tar_len: int):
   midi_ds = (tf.data.Dataset
      .from_tensor_slices(filenames)
      .map(lambda x: tf.py_function(_process_raw_midi, inp=[x], Tout=tf.string), 
           num_parallel_calls=tf.data.AUTOTUNE))
   all_tokens = (token for sequence in midi_ds.as_numpy_iterator() for token in sequence)
   midi_tokenizer = tokenizer.MidiTokenizer(all_tokens)
   midi_ds = (midi_ds
      .map(lambda x: _tokenize_and_split(midi_tokenizer, x, inp_len, tar_len))
      .padded_batch(batch_size))
   return _optimize_dataset(midi_ds), midi_tokenizer

def load(midi_path: str, *, 
         train_size: float, 
         val_size: float, 
         test_size: float, 
         batch_size: int, 
         inp_len: int, 
         tar_len: int):
   """
   returns a 3-tuple of `tf.Dataset` each returning `(input_seq, target_seq)`, representing train, 
   validation, and test portions of the overall dataset. `input_seq` represents the `inp_split` 
   portion of each midi sequence in `midi_path`.
   """
   # get midi files
   filenames = tf.random.shuffle(glob.glob(f'{midi_path}/**/*.midi', recursive=True))

   # get train, validation, and test sizes
   train_split, val_split, _ = _get_dataset_splits(train_size, val_size, test_size)
   train_split = int(train_split * len(filenames))
   val_split = int(val_split * len(filenames))
   
   # split filenames to train, test, split
   midi_ds, midi_tokenizer = _create_dataset(filenames, batch_size, inp_len, tar_len)
   train_ds = midi_ds.take(train_split)
   val_ds = train_ds.skip(train_split)
   test_ds = val_ds.skip(val_split)
   val_ds = val_ds.take(val_split)

   return train_ds, val_ds, test_ds, midi_tokenizer
