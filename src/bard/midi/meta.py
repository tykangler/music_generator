import tensorflow as tf
from .convert import Wait

def epoch(complete_name: str, composers_json: list):
   result = [composer for composer in composers_json 
             if composer['complete_name'].lower() == complete_name.lower()]
   if len(result) > 0:
      return result[0]['epoch']
   return None

def find_tempo(it):
   DEFAULT_VALUE = 500_000
   for x in it:
      if x.is_meta and x.type == 'set_tempo':
         return x.tempo
   return DEFAULT_VALUE

def _average(it):
   sum = 0
   len = 0
   for x in it:
      sum += x
      len += 1
   return sum / len

def _scale_exp(x):
   return tf.exp(x / 500)

def average_wait_time(it, ticks_per_beat, tempo):
   microseconds_per_tick = tempo / ticks_per_beat
   avg_wait = _average(x.duration() for x in it if isinstance(x, Wait)) * microseconds_per_tick 
   return _scale_exp(avg_wait)
