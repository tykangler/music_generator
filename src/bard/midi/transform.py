import numpy as np

# tokenizer --

class Tokenizer:
   """
   encodes midi vocabulary strings (i.e. `note_p:10_v:2_i:piano`) to unique integers
   """
   def __init__(self):
      pass

# encoder --

class Stringify:
   """
   Encodes midi objects to strings in vocabulary. midi objects are converted into two messages:
      * `rest_b:{beats}`
      * `[note_p:{pitch}_v:{velocity}_i:{instrument}|control_c:{control}_v:{value}]`
   """
   def __init__(self, resolution):
      self.resolution = resolution

   def _rest(self, beats: int):
      return f'rest_b:{beats}'

   def _note(self, pitch: int, velocity: int, instrument: str='piano'):
      return f'note_p:{pitch}_v:{velocity}_i:{instrument}'

   def _control(self, control: int, value: int):
      return f'control_c:{control}_v:{value}'

   _MESSAGE_REF = {
      'note_on': _note,
      'note_off': _note,
      'control_change': _control
   }

   def _get_message(self, msg):
      """
      constructs simple two byte messages (Control and Note). Doesn't 
      handle instruments. Doesn't handle _note offs with non zero velocity
      """
      msg_data = msg.bytes()[1:]
      if msg.type == 'note_off':
         msg_data[-1] = 0
      msg_obj = self._MESSAGE_REF.get(msg.type, None)
      return msg_obj(*msg_data) if msg_obj is not None else msg_obj

   def encode_midi_to_vocab(self, seq):
      def gen_messages(seq):
         accum_beats = 0
         for msg in seq:
            accum_beats += msg.time
            msg_obj = self._get_message(msg)
            if msg_obj is not None:
               excess_beats = accum_beats % self.resolution
               for i in range(accum_beats // self.resolution):
                     yield self._rest(self.resolution)
               if excess_beats != 0:
                     yield self._rest(excess_beats)
               accum_beats = 0
               yield msg_obj
      return list(gen_messages(seq))

   def __call__(self, inputs):
      inputs.tracks[1] = [msg for msg in inputs.tracks[1] if msg.type in self._MESSAGE_REF]
      final =  self.encode_midi_to_vocab(inputs.tracks[1])
      return final

# quantize --------

class TimeQuantizer:
   """
   quantizes time attribute according to resolution.
   """
   def __init__(self, resolution):
      self.resolution = resolution

   def _nearest_mult(val, multiple):
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

      quantized_times = self.quantize_to_beats([msg.time for i, msg in enumerate(main_track) if i % 2 == 0], ticks_per_beat)
      inputs.tracks[1] = [msg.copy(time=new_time) for msg, new_time in zip(inputs, quantized_times)]
      return inputs

# bin ----------

class VelocityBinner:
   def __init__(self, min_val, max_val, num_bins):
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
      # num_bins + 1 to create 32 intervals
      return np.searchsorted(bins, sequence, side='right')

   def __call__(self, inputs):
      note_types = { 'note_on', 'note_off' }
      binned_velocities = self.bin_seq([msg.velocity for msg in inputs.tracks[1] 
                                        if msg.type in note_types])
      velocity_iter = iter(binned_velocities)
      inputs.tracks[1] = [msg.copy(velocity=next(velocity_iter)) 
                          if msg.type in note_types else msg.copy()
                          for msg in inputs]
      return inputs
