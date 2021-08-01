import pytest
import os
import numpy as np
import mido

from bard.midi import transform

RESOLUTION = 32
MIN_VELOCITY_VAL = 1
MAX_VELOCITY_VAL = 128
NUM_VELOCITY_BINS = 32

DATA_DIR = os.path.abspath('../../../../data/maestro-v3.0.0/2004')

TICKS_PER_BEAT = 320
MAX_TIME = 1000

def make_converter():
   return transform.VocabularyConverter(RESOLUTION)

def make_quantizer():
   return transform.TimeQuantizer(RESOLUTION)

def make_binner():
   return transform.VelocityBinner(MIN_VELOCITY_VAL, MAX_VELOCITY_VAL, NUM_VELOCITY_BINS)

def get_midi():
   midifile = mido.MidiFile()
   midifile.ticks_per_beat = TICKS_PER_BEAT
   midifile.add_track()
   midifile.add_track()
   return midifile

def test_quantizer_produces_correct_times():
   # set up midifile
   quantizer = make_quantizer()
   midifile = get_midi()
   for i in range(MAX_TIME):
      midifile.tracks[1].append(mido.Message('note_on', time=i))

   # build expected times
   tick_res = TICKS_PER_BEAT // RESOLUTION # tick_res = 10
   expected_times = [0] * 5
   for i in range(1, MAX_TIME // tick_res):
      expected_times += [i] * 10 # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, ...]
   expected_times += [MAX_TIME // tick_res] * 5 # [..., 100, 100, 100, 100, 100]

   # quantize midifile
   midifile = quantizer(midifile)
   for i, msg in enumerate(midifile.tracks[1]):
      print(msg.time, expected_times[i])
      assert msg.time == expected_times[i]

def test_binner_produces_correct_values_for_nonzero_velocity():
   binner = make_binner()
   bin_size = (MAX_VELOCITY_VAL - MIN_VELOCITY_VAL) / NUM_VELOCITY_BINS
   midifile = get_midi()
   for i in range(MIN_VELOCITY_VAL, MAX_VELOCITY_VAL):
      midifile.tracks[1].append(mido.Message('note_on', velocity=i))

   note_types = {'note_on', 'note_off'}
   midifile = binner(midifile)
   for i, msg in enumerate(midifile.tracks[1]):
      assert msg.velocity == int(i // bin_size + MIN_VELOCITY_VAL) if msg.type in note_types else True

def test_binner_keeps_all_messages():
   binner = make_binner()
   midifile = get_midi()
   for i in range(100):
      midifile.tracks[1].append(mido.Message('note_on'))
      midifile.tracks[1].append(mido.Message('note_off'))
      midifile.tracks[1].append(mido.Message('control_change'))

   num_msgs = len(midifile.tracks[1])
   midifile = binner(midifile)
   assert len(midifile.tracks[1]) == num_msgs
