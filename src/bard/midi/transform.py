import numpy as np
      
def rest(beats: int):
   return f'rest_b:{beats}'

def note(pitch: int, velocity: int, instrument: str='piano'):
   return f'note_p:{pitch}_v:{velocity}_i:{instrument}'

def control(control: int, value: int):
   return f'control_c:{control}_v:{value}'

_MESSAGE_REF = {
   'note_on': note,
   'note_off': note,
   'control_change': control
}

def _get_message(msg):
   """
   constructs simple two byte messages (Control and Note). Doesn't 
   handle instruments. Doesn't handle note offs with non zero velocity
   """
   msg_data = msg.bytes()[1:]
   if msg.type == 'note_off':
      msg_data[-1] = 0
   msg_obj = _MESSAGE_REF.get(msg.type, None)
   return msg_obj(*msg_data) if msg_obj is not None else msg_obj

# export
def encode_midi_to_vocab(track, tick_per_beat, resolution):
   def gen_messages(track, tick_per_beat, resolution):
      accum_beats = 0
      for msg in track:
         accum_beats += msg.time
         msg_obj = _get_message(msg)
         if msg_obj is not None:
            excess_beats = accum_beats % resolution
            for i in range(accum_beats // resolution):
                  yield rest(resolution)
            if excess_beats != 0:
                  yield rest(excess_beats)
            accum_beats = 0
            yield msg_obj
   return list(gen_messages(track, tick_per_beat, resolution))

# quantize --------

def _nearest_mult(val, multiple):
   temp = val + multiple / 2
   return temp - temp % multiple

def quantize_to_beats(sequence, ticks_per_beat: int, resolution: int, seqlen: int=None) -> np.ndarray:
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
   seqlen = len(sequence) if seqlen is None else seqlen
   tick_res = ticks_per_beat / resolution

   quantized = (int(_nearest_mult(val, tick_res)) * resolution // ticks_per_beat 
                for val in sequence)
   return np.fromiter(quantized, dtype=np.int32, count=seqlen)

# bin ----------

def _get_bin(value, bin_size: int, max_val: int, min_val: int) -> int:
   if value >= max_val:
      return bin_size + 1
   elif value < min_val:
      return 0
   else:
      return int((value - min_val) // bin_size) + 1

def bin_seq(sequence, num_bins: int, max_val: int, min_val: int, seqlen: int=None) -> np.ndarray:
   """
   takes in an iterable of numbers and assigns an integer label corresponding to the bin it falls
   in. Values greater than max_val are given a bin label of num_bins + 1, and values below min_val
   are given bin label 0.
   """
   bin_size = (max_val - min_val) / num_bins
   seqlen = len(sequence) if seqlen is None else seqlen

   binned = (_get_bin(val, bin_size, max_val, min_val) for val in sequence)
   return np.fromiter(binned, dtype=np.int32, count=seqlen)
