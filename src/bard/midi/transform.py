import numpy as np

from . import vocabulary

# encoder --

class VocabularyConverter:
   """
   Encodes midi objects to tokens in vocabulary.
   """
   def __init__(self, resolution):
      self.resolution = resolution

   def __call__(self, inputs):
      relevant_midi = [msg for msg in inputs.tracks[1] if msg.type in vocabulary.relevant_messages]
      return list(vocabulary.encode(relevant_midi, self.resolution))

# quantize --------

class TimeQuantizer:
   """
   quantizes time attribute according to resolution.
   """
   def __init__(self, resolution):
      self.resolution = resolution

   def _nearest_mult(self, val, multiple):
      temp = val + multiple / 2
      return temp - temp % multiple

   def quantize_to_beats(self, sequence, ticks_per_beat: int) -> np.ndarray:
      """
      takes in an iterable of integer values representing ticks, and snaps to the nearest resolution
      multiple. The new tick values are then converted to beats, following equation
      `beats = ticks / (ticks per beat)`. i.e. 48 ticks, 384 tpb = 0.125 beats. This value is then
      expressed in terms of `resolution` so that the final value is `x = beats * resolution`, or
      `x = (ticks * resolution) / (ticks per beat)`.

      params:
         sequence: sequence to quantize
         ticks_per_beat: ticks per beat as given in midi meta messages
         resolution: desired quantization resolution
         key: optional callable to be called on each element in the sequence before binning
      returns:
         sequence of integers with original values snapped to nearest multiple
      """
      tick_res = ticks_per_beat / self.resolution

      quantized = [int(self._nearest_mult(val, tick_res)) * self.resolution // ticks_per_beat
                  for val in sequence]
      return quantized

   def __call__(self, inputs):
      assert hasattr(inputs, 'ticks_per_beat')
      ticks_per_beat = inputs.ticks_per_beat
      main_track = inputs.tracks[1]

      quantized_times = self.quantize_to_beats([msg.time for msg in main_track], ticks_per_beat)
      inputs.tracks[1] = [msg.copy(time=new_time) for msg, new_time in zip(inputs, quantized_times)]
      return inputs

# bin ----------

class VelocityBinner:
   def __init__(self, min_val, max_val, num_bins):
      """total number of bins is num_bins + 2 for values above and below max_val and min_val
      respectively"""
      self.min_val = min_val
      self.max_val = max_val
      self.num_bins = num_bins

   def bin_seq(self, sequence) -> np.ndarray:
      """
      takes in an iterable of integeres (typically velocity) and assigns an integer label
      corresponding to the bin it falls in. Values greater than max_val are given a bin label of
      num_bins + 1, and values below min_val are given bin label 0.
      """
      bins = np.linspace(start=self.min_val, stop=self.max_val, num=self.num_bins + 1)
      # num_bins + 1 to create 32 main intervals
      return np.searchsorted(bins, sequence, side='right')

   def __call__(self, inputs):
      note_types = { 'note_on', 'note_off' }
      main_track = inputs.tracks[1]
      binned_velocities = self.bin_seq([msg.velocity for msg in main_track
                                        if msg.type in note_types])
      velocity_iter = iter(binned_velocities)
      inputs.tracks[1] = [msg.copy(velocity=next(velocity_iter))
                          if msg.type in note_types else msg.copy()
                          for msg in main_track]
      return inputs
