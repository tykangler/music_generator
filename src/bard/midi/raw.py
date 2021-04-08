import numpy as np
import tensorflow as tf

from .messages import *

class Preprocessor:
   """
   A pipeline of steps to process input data through. Each step must be a callable taking in
   a midi sequence as its sole argument. After each step has been run, the processed data is
   converted into a sequence of strings representing either 'note', 'control', or 'rest'
   """
   _MESSAGE_REF = {
      'note_on': note,
      'note_off': note,
      'control_change': control
   }
   
   def __init__(self, steps=None):
      self.steps: list = [] if steps is None else steps
   
   def add(self, step):
      self.steps.append(step)
   
   def pop(self):
      self.steps.pop()

   def __call__(self, inputs):
      for step in self.steps:
         inputs = step(inputs)
      obj_encoded = list(self._encode(inputs))
      return obj_encoded

   def _encode(self, track) -> np.ndarray:
      """
      takes in a sequence (midi track), and returns an iterator to a vocabularized sequence 
      with instances of Rest, Note, and Control. Timings are separated from each midi event and 
      made their own entity to signal no event has occurred.
      """
      accum_time = 0
      for msg in track:
         accum_time += msg.time
         msg_obj = self._get_message(msg)
         if msg_obj is not None:
            yield rest(accum_time)
            yield msg_obj
   
   def _get_message(self, msg):
      """
      constructs simple two byte messages (Control and Note). Doesn't 
      handle instruments. Doesn't handle note offs with varying velocity.
      """
      msg_data = msg.bytes()[1:]
      if msg.type == 'note_off':
         msg_data[-1] = 0
      msg_obj = self._MESSAGE_REF.get(msg.type, None)
      return msg_obj(*msg_data) if msg_obj is not None else msg_obj

# setting fields:
# to bin/quantize: filter out, grab fields, pass to, result -> [1, 2, 0, 4, 0, 31]. 
# from bin/quantize: msg.copy(field=new_field) for msg in res 
