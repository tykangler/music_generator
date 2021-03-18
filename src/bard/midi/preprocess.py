import numpy as np
import dataclasses

@dataclasses.dataclass
class Control:
   """
   A controller event in a midi sequence.
   params:
      control: integer value of the activated controller
      value: value the controller is set to
   """
   control: int
   value: int

   def __hash__(self):
      return hash((self.control, self.value))

@dataclasses.dataclass
class Note:
   """
   A note event in a midi sequence.
   params:
      pitch: integer value greater than or equal to 0
      velocity: integer value greater than or equal to 0. If 0, indicates no sound.
      instrument: the instrument playing the note
   """
   pitch: int
   velocity: int
   instrument: str = 'piano'

   def __hash__(self):
      return hash((self.pitch, self.velocity, self.instrument))

@dataclasses.dataclass
class Wait:
   """
   Indicates that no event has occurred. Acts as a record of deltatime.
   params:
      beats: duration of wait in number of beats
   """
   beats: int

   def __hash__(self):
      return hash(self.beats)

# function definitions

def _nearest_mult(val, multiple):
   temp = val + multiple / 2
   return temp - temp % multiple

def quantize(sequence, ticks_per_beat: int, resolution: int, seqlen: int=None, key=None) -> np.ndarray:
   """
   snaps values in the given sequence so that each value is replaced with an integer in the range
   [0, resolution). i.e. a multiple of resolution
   params:
      sequence: sequence to quantize,
      ticks_per_beat: ticks per beat as given in midi metadata
      resolution: desired quantization resolution
      key: optional callable to be called on each element in the sequence before binning
   """
   if key is None:
      key = lambda x: x
   seqlen = len(sequence) if seqlen is None else seqlen
   tick_res = ticks_per_beat / resolution
   quantized = (int(_nearest_mult(key(elem), tick_res) / tick_res) for elem in sequence)
   return np.fromiter(quantized, dtype=np.int32, count=seqlen)

def vocabularize(sequence, seqlen=None) -> np.ndarray:
   """
   takes in a sequence (midi track) and vocabualarizes sequence into instances of Wait, Note,
   and Control. Timings are separated from each midi event and made their own entity to signal
   no event has occurred.
   """
   vocabularized = ([Wait(msg.time), Note(msg.note, msg.velocity) 
      if msg.type == 'note_on' else Control(msg.control, msg.value)] 
      for msg in sequence)
   seqlen = len(sequence) if seqlen is None else seqlen
   vocab_track: np.ndarray = np.fromiter(vocabularized, dtype=np.int32, count=seqlen)
   return vocab_track.ravel()[1:]

def get_bin_label(value, bin_size: int, max_val: int, min_val: int) -> int:
   if (value >= max_val):
      return bin_size + 1
   elif (value < min_val):
      return 0
   else:
      return int(value // bin_size) + 1


def bin(sequence, num_bins: int, max_val: int, min_val: int=0, seqlen: int=None, key=None) -> np.ndarray:
   """
   returns sequence with labels corresponding to ranges for each respective element. Bin sizes
   are calculated as (max_val - min_val) / num_bins. An extra two bins (labels 0, and num_bins + 1)
   for elements that are outside the given range [min_val, max_val)
   """
   bin_size = (max_val - min_val) / num_bins
   seqlen = len(sequence) if seqlen is None else seqlen
   binned = (get_bin_label(key(elem), bin_size, max_val, min_val) for elem in sequence)
   return np.from_iter(binned, dtype=np.int32, count=seqlen)
