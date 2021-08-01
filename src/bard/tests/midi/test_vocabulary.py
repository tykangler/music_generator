import pytest
import numpy as np
from mido import Message

from bard.midi import vocabulary

RESOLUTION = 32

# helper functions ---

def note_on(time=0, note=-1, velocity=-1):
   if note == -1:
      note = np.random.randint(1, 128)
   if velocity == -1:
      velocity = np.random.randint(1, 128)
   return Message('note_on', note=note, velocity=velocity, time=time)

def note_off(time=0, note=-1, velocity=-1):
   if note == -1:
      note = np.random.randint(1, 128)
   if velocity == -1:
      velocity = np.random.randint(1, 128)
   return Message('note_off', note=note, velocity=velocity, time=time)

def control(time=0, control=-1, value=-1):
   if control == -1:
      control = np.random.randint(1, 128)
   if value == -1:
      value = np.random.randint(1, 128)
   return Message('control_change', control=control, value=value, time=time)

# encode decode msg ---

@pytest.mark.parametrize("type, inputs, expected,", [
   (vocabulary.NOTE, [1, 2, 3], 0b10_0000_0001_0000_0010_0000_0011),
   (vocabulary.NOTE, [20, 127, 79], 0b10_0001_0100_0111_1111_0100_1111),
   (vocabulary.CONTROL, [100, 0], 0b11_0110_0100_0000_0000),
   (vocabulary.REST, [100], 0b1_0110_0100)
])
class TestMsgEncoderDecoder:
   def test_encode_msg_generates_correct_output(self, type, inputs, expected):
      encoded = vocabulary.encode_msg(type, inputs)
      assert encoded == expected

   def test_decode_msg_produces_original_msg(self, type, inputs, expected):
      encoded = vocabulary.encode_msg(type, inputs)
      decoded = vocabulary.decode_msg(encoded)
      assert decoded == (type, inputs)

# encode decode seq ---

## encode ---

@pytest.mark.parametrize("inputs", [
   [note_on(), note_on(), note_on(), note_on(), note_off(), note_off(),
    control(), control(), note_off()]
])
def test_encode_zero_time_should_have_no_rests(inputs):
   encoded = list(vocabulary.encode(inputs, RESOLUTION))
   assert len(inputs) == len(encoded)

def test_encode_note_on_zero_velocity_and_note_off_same_result():
   for note in range(128):
      encoded_note_on = list(vocabulary.encode([note_on(note=note, velocity=0)], RESOLUTION))
      encoded_note_off = list(vocabulary.encode([note_off(note=note)], RESOLUTION))
      assert encoded_note_on[0] == encoded_note_off[0]

def test_encode_with_rests_within_resolution():
   inputs = [
      note_on(20), note_off(10), note_on(22), control(0), note_on(0), note_on(30),
      note_off(12)
   ]
   encoded = list(vocabulary.encode(inputs, RESOLUTION))
   assert len(encoded) == len(inputs) + 5

@pytest.mark.parametrize("inputs, num_rests", [
   ([note_on(20), note_off(33), note_on(40), control(33), control(0), note_off(12)], 8),
   ([note_on(33), note_off(20), note_on(64), note_off(65), control(0), note_off(22)], 9),
   ([note_on(4), note_off(32), control(2), note_on(0), control(0)], 3)
])
def test_encode_with_rests_more_than_resolution(inputs, num_rests):
   encoded = list(vocabulary.encode(inputs, RESOLUTION))
   assert len(encoded) == len(inputs) + num_rests

## decode ---

@pytest.mark.parametrize("inputs", [
   [note_on(12), note_off(0), control(33), note_on(23), control(12), note_off(15)],
   [note_on(64), note_off(32), note_on(0), control(0), note_on(46), note_off(66)],
   [note_on(), note_off(), control(), note_on(), note_off()],
   [note_on(23), note_off(12), note_on(2), note_off(23), note_on(30)]
])
def test_decode_produces_original_input(inputs):
   encoded = vocabulary.encode(inputs, RESOLUTION)
   for decoded, original in zip(vocabulary.decode(encoded), inputs):
      if original.type == 'note_off':
         assert decoded.bytes()[1:-1] == original.bytes()[1:-1]
      else:
         assert decoded.bytes() == original.bytes()
