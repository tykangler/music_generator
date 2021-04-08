def rest(beats: int):
   return f'rest_b:{beats}'

def note(pitch: int, velocity: int, instrument: str='piano'):
   return f'note_p:{pitch}_v:{velocity}_i:{instrument}'

def control(control: int, value: int):
   return f'control_c:{control}_v:{value}'
