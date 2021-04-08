import numpy as np

# quantize --------

def _nearest_mult(val, multiple):
   temp = val + multiple / 2
   return temp - temp % multiple

def quantize(sequence, ticks_per_beat: int, resolution: int, seqlen: int=None) -> np.ndarray:
   """
   takes in an iterable of integers and snaps values to the nearest resolution multiple,
   so that each value is replaced with an integer in the range [0, resolution)
   params:
      sequence: sequence to quantize
      ticks_per_beat: ticks per beat as given in midi metadata
      resolution: desired quantization resolution
      key: optional callable to be called on each element in the sequence before binning
   returns:
      sequence of integers with original values snapped to nearest multiple
   """
   seqlen = len(sequence) if seqlen is None else seqlen
   tick_res = ticks_per_beat / resolution

   quantized = (int(_nearest_mult(val, tick_res) / tick_res) for val in sequence)
   return np.fromiter(quantized, dtype=np.int32, count=seqlen)

# bin ----------

def _get_bin(value, bin_size: int, max_val: int, min_val: int) -> int:
   if value >= max_val:
      return bin_size + 1
   elif value < min_val:
      return 0
   else:
      return int(value // bin_size) + 1

def bin(sequence, num_bins: int, max_val: int, min_val: int, seqlen: int=None) -> np.ndarray:
   """
   takes in an iterable of numbers and assigns an integer label corresponding to the bin it falls
   in. Values greater than max_val are given a bin label of num_bins + 1, and values below min_val
   are given bin label 0.
   """
   bin_size = (max_val - min_val) / num_bins
   seqlen = len(sequence) if seqlen is None else seqlen

   binned = (_get_bin(val, bin_size, max_val, min_val) for val in sequence)
   return np.from_iter(binned, dtype=np.int32, count=seqlen)
