
def get_pitch(note: int):
   notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
   return f"{notes[note % 12]}{note // 12 - 1}"

def is_note_on(msg):
   return msg.type == 'note_on' and msg.velocity > 0

def is_note_off(msg):
   return (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off'
