#!/usr/bin/env python
# coding: utf-8

# # Generating music

# To generate music, I will use the **transformer** model, used by OpenAI in their GPT-2 model. In prior iterations, I had considered using RNNs with the seq2seq model. But after further research, I discovered that the transformer model can achieve better performance and accuracy, with the added benefit of parallelization, albeit at the cost of memory. 

# In[1]:


import mido
import os
import math
import matplotlib.pyplot as plt
from itertools import chain, islice
from functools import cmp_to_key
from copy import deepcopy
from fractions import Fraction
import json


# ## Data Preparation

# In[2]:


DATA_DIR = '../../data'


# In[3]:


def get_pitch(note: int):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{notes[note % 12]}{note // 12 - 1}"


# The Test Score: <br>
# <img height=800 width=600 src="https://imslp.org/images/8/8b/TN-Schumann%2C_Robert_Werke_Breitkopf_Gregg_Serie_7_Band_2_RS_51_Op_13_scan.jpg"></img> <br>
# Robert Schumann,Symphonic Etudes Op. 13 (with Posthumous variations)
# 

# In[4]:


# define function to play music
TEST_FILE = f'{DATA_DIR}/maestro-v3.0.0/2018/MIDI-Unprocessed_Recital20_MID--AUDIO_20_R1_2018_wav--4.midi'
def play_midi(file_name=TEST_FILE):
    os.startfile(os.path.abspath(file_name))


# In[5]:


is_note_on = lambda msg: msg.type == 'note_on' and msg.velocity > 0
is_note_off = lambda msg: (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off'


# In[6]:


# make sure to iterate over tracks for correct deltatime value (ticks), otherwise deltatime is seconds. This is also faster.
def print_messages(track, limit: int=0, *, transnote=None, transvelocity=None, transtime=None):
    transnote = transnote or (lambda note: note) 
    transvelocity = transvelocity or (lambda velocity: velocity)
    transtime = transtime or (lambda time: time)
    track = islice(track, limit) if limit > 0 else track
    for event in track:
        if is_note_on(event):
            note, velocity, time = transnote(event.note), transvelocity(event.velocity), transtime(event.time)
            print(f'note_on channel={event.channel} note={note} velocity={velocity} time={time}')
        elif is_note_off(event):
            note, velocity, time = transnote(event.note), transvelocity(event.velocity), transtime(event.time)
            print(f'note_off channel={event.channel} note={note} time={time}')
        else:
            print(event)


# In[7]:


schumann = mido.MidiFile(TEST_FILE)


# In[8]:


schumann.ticks_per_beat


# In[9]:


print_messages(schumann.tracks[1], 20, transnote=get_pitch)


# ### Converting Ticks to Beats
# 
# Time attribute represents <deltatime\>. <deltatime\> is represented as number of ticks before playing the message. The number of ticks per beat is defined in the MThd chunk as <division\>. (i.e. <division\> = 96 means 96 ticks per beat). The number of microseconds per beat is defined as $500,000 \frac{\mu s}{beat}$, or can be set in the meta message 'set_tempo' in each track.
# 
# So $time = 288$, $division = 384$, $tempo = 500,000$ equates to $\frac{500,000}{384} * 288 = 375,000 \mu s$
# 
# This is the ticks between the first note_on (C#5) and the corresponding note off
# 
# This is equivalent to .375 seconds, which is the deltatime value when using `midi.play()` or `iter(midi)`

# 
# 
#  With $500,000 \frac{\mu s}{beat}$, BPM = 120. The denominator of the time signature tells what kind of note (quarter, eighth) is a beat. The numerator tells how many beats are in bar. With a time signature of 4/4, a beat is a quarter note. 
# 
# With $time = 288$, and $division = 384$, $288 \ \text{ticks} * \frac{1}{384} \frac{beat}{tick} = .75 \ \text{beats}$
# 
# This is equal to $375,000 \mu s * \frac{1}{500000} \frac{beat}{\mu s} = .75 \ \text{beats}$.
# 
# A time signature of 4 means $time = 288$ is .75 of a quarter note. However, from the image above, the first notes are quarter notes, not fractions of quarter notes, probably because the performance was played with a different BPM in mind (Andante, maybe 90), and not the one given in the midi file. 
# 

# ### Quantization
# 
# Quantize notes so that notes will have deltatime corrected to the nearest multiple of $\epsilon$. A lower $\epsilon$ means a higher frequency, but also more off-beats. A greater $\epsilon$ means lower frequency, and more synchronization.

# In[10]:


BEAT_RESOLUTION = 64
BEAT = 4


# In[11]:


def nearest_tick(ticks, resolution):
    temp = ticks + resolution // 2
    return int(temp - temp % resolution)

def quantize(track, resolution, ticks_per_beat, beat=4, limit: int=0):
    tick_resolution = beat * ticks_per_beat / resolution
    if limit > 0:
        track = islice(track, limit)
    return [msg.copy(time=nearest_tick(msg.time, tick_resolution)) 
            if msg.type == 'note_on' else msg.copy() 
            for msg in track]

quantized = quantize(schumann.tracks[1], resolution=BEAT_RESOLUTION, beat=BEAT, ticks_per_beat=schumann.ticks_per_beat)
print_messages(quantized, 20, transnote=get_pitch)


# In[12]:


schumann_copy = deepcopy(schumann)
schumann_copy.tracks[1] = mido.MidiTrack(quantized)


# In[13]:


schumann_copy.save('schumann.midi')


# ## Moving Away from Midi

# In[14]:


# class setup
class Note:
    def __init__(self, pitch, velocity, instrument='piano'):
        self.pitch = pitch
        self.velocity = velocity
        self.instrument = instrument

    def __str__(self):
        return f"{self.pitch}:{self.velocity}:{self.instrument}"

    def __repr__(self):
        return f"<Note pitch={self.pitch} velocity={self.velocity} instrument={self.instrument}>"

    def __eq__(self):
        return self.pitch == other.pitch and self.velocity == other.velocity and self.instrument == other.instrument

class Wait:
    def __init__(self, beats):
        self.beats = beats

    def duration(self):
        return self.beats.numerator
    
    def __str__(self):
        return f"wait:{self.beats.numerator}"
    
    def __repr__(self):
        return f"<Wait {self.beats}>"
    
    def __eq__(self, other):
        return self.beats == other.beats


# ### Converting to Notes

# In[15]:


def to_beats(track, tick_per_beat, limit=0):
    if limit > 0:
        track = islice(track, limit)
    tick_to_beat = lambda tick, tick_per_beat: tick / tick_per_beat 
    return [(Wait(Fraction(tick_to_beat(msg.time, tick_per_beat))), Note(get_pitch(msg.note), msg.velocity))
            for msg in track if msg.type == 'note_on']


# In[16]:


schumann_seq = to_beats(schumann_copy.tracks[1], schumann_copy.ticks_per_beat)
schumann_seq[:30]


# The last value in each tuple represents the number of beats for that note is played. Because notes were quantized with a 64th note resolution and beats set to quarter notes, each note will have a time that is some multiple of $\frac{1}{16}$. 
# 
# > When quantizing, each tick value was adjusted so that $$\text{ticks}' = x * \frac{\text{beat} * \text{ticks_per_beat}}{\text{resolution}}$$When converting to beats, $\text{ticks}'$ is divided by ticks_per_beat. $$\text{beats} = \frac{\text{ticks}'}{\text{ticks_per_beat}} = x * \frac{\text{beat} * \text{ticks_per_beat}}{\text{resolution} * \text{ticks_per_beat}} = x * \frac{\text{beat}}{\text{resolution}}$$

# In[17]:


BEAT_BASE = BEAT / BEAT_RESOLUTION


# Using this approach, we will have 16 waits defined in the vocabulary (wait:[1-16]). However, a constant resolution needs to be set for all songs. Test quantizing a faster song.

# ### Quantizing faster songs

# In[18]:


TEST_FAST_FILE = f'{DATA_DIR}/maestro-v3.0.0/2004/MIDI-Unprocessed_XP_08_R1_2004_04-06_ORIG_MID--AUDIO_08_R1_2004_05_Track05_wav--2.midi'
rachmaninoff = mido.MidiFile(TEST_FAST_FILE)
print_messages(rachmaninoff.tracks[1], 20)


# In[19]:


rachmaninoff_copy = deepcopy(rachmaninoff)
rachmaninoff_copy.tracks[1] = quantize(rachmaninoff_copy.tracks[1], 64, rachmaninoff_copy.ticks_per_beat)
print_messages(rachmaninoff_copy.tracks[1], limit=20)


# In[20]:


rachmaninoff_copy.save('rachmaninoff.midi')


# In[21]:


to_beats(rachmaninoff_copy.tracks[1], rachmaninoff_copy.ticks_per_beat)[:30]


# Quantizing a faster song works fine. Quantization performance depends on the ticks per beat and the resolution. As long as ticks_per_beat is fairly small (so that tick_resolution is small), then overall resolution will be retained. In practice, we can scale down ticks_per_beat, and <set_tempo\> by the same amount. Scaling both down will keep $\frac{\mu s}{tick}$ the same, so that the absolute timings of each message are the same.

# ### Inserting Waits

# In[22]:


def with_waits(beats_stream):
    first = True
    for wait, note in beats_stream:
        if not first and wait.duration() != 0:
            yield wait
        first = False
        yield note


# In[23]:


for i in islice(with_waits(schumann_seq), 20):
    print(i)


# ### Additional Features
# 
# Aside from the vocabulary, I will train the model with the following features:
# 
# - [x] composer
# - ~~key signature~~ (data not available)
# - [x] tempo
# - [x] time period/style
# 

# In[24]:


with open(f'{DATA_DIR}/metadata/composers.json') as composer_file:
    composers = json.load(composer_file)


# For composer searching, I will need to clean up the csv file to make sure that composer names match up with the names given in the composers file.

# In[25]:


def epoch(complete_name: str):
    result = [composer for composer in composers if composer['complete_name'].lower() == complete_name.lower()]
    if len(result) > 0:
        return result[0]['epoch']
    return None


# In[26]:


epoch('Leoš Janáček')


# In[27]:


def find_tempo(it):
    DEFAULT_VALUE = 500_000
    for x in it:
        if x.is_meta and x.type == 'set_tempo':
            return x.tempo
    return DEFAULT_VALUE

def average(it):
    sum = 0
    len = 0
    for x in it:
        sum += x
        len += 1
    return sum / len

def scale_exp(x):
    return math.exp(x / 500)

def average_wait_time(it, ticks_per_beat, tempo):
    microseconds_per_tick = tempo / ticks_per_beat
    avg_wait = average(x.duration() for x in it if isinstance(x, Wait)) * microseconds_per_tick 
    return scale_exp(avg_wait)


# In[28]:


print('schumann:')
print(average_wait_time(with_waits(schumann_seq), schumann_copy.ticks_per_beat, find_tempo(schumann_copy.tracks[0])))
print('rachmaninoff:')
print(average_wait_time(
    with_waits(to_beats(rachmaninoff_copy.tracks[1], rachmaninoff_copy.ticks_per_beat)), 
    rachmaninoff_copy.ticks_per_beat, 
    find_tempo(rachmaninoff_copy.tracks[0])))


# Approximation I'll use to measure tempo, where MSPT = microseconds per tick:
# $$\tau = \text{exp}(\frac{MSPT * \overline{\text{wait_in_ticks}}}{500})$$

# ## Vocabulary

# The full vocabulary will consist of 16 waits (or any power of 2 depending on what resolution I go with), and 128 notes, each with 32 volume levels, each with 6 instruments. A total size of 24,592.
# 
# Volumes will be binned for less noise, and to account for variability in performances.
# 
# For now I'll just stick with piano, so 88 notes (A0-C8), 32 volumes, 1 instrument, 16 waits. A total size of 2832. 

# ## Test playback

# In[29]:


# Test playback original
play_midi()


# In[30]:


# Test playback quantized
play_midi('schumann.midi')


# In[31]:


# Test playback original (fast)
play_midi(TEST_FAST_FILE)


# In[32]:


# Test playback quantized (fast)
play_midi('rachmaninoff.midi')

