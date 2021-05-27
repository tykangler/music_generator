def _rest(beats: int):
   return f'rest_b:{beats}'

def _note(pitch: int, velocity: int, instrument: str='piano'):
   return f'note_p:{pitch}_v:{velocity}_i:{instrument}'

def _control(control: int, value: int):
   return f'control_c:{control}_v:{value}'

relevant_messages = {
   'note_on': _note,
   'note_off': _note,
   'control_change': _control
}

def _get_message(msg):
   """
   constructs simple two byte messages (Control and Note). Doesn't 
   handle instruments. Doesn't handle note offs with non zero velocity
   """
   msg_data = msg.bytes()[1:]
   if msg.type == 'note_off':
      msg_data[-1] = 0
   msg_obj = relevant_messages.get(msg.type, None)
   return msg_obj(*msg_data) if msg_obj is not None else msg_obj

def encode(seq, resolution):
   accum_beats = 0
   for msg in seq:
      accum_beats += msg.time
      msg_obj = _get_message(msg)
      if msg_obj is not None:
         excess_beats = accum_beats % resolution
         for i in range(accum_beats // resolution):
            yield _rest(resolution)
         if excess_beats != 0:
            yield _rest(excess_beats)
         accum_beats = 0
         yield msg_obj
