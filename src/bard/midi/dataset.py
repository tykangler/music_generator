import os
import csv
import json

import mido
import numpy as np

def get_epoch(complete_name: str, composers_json: list):
   """
   returns the epoch of the composer given in `complete_name`, from `composers_json`
   """
   result = [composer for composer in composers_json 
             if composer['complete_name'].lower() == complete_name.lower()]
   if len(result) > 0:
      return result[0]['epoch']
   return None

def get_composer(file_path: str, csv_metadata: list):
   """
   returns the name of the composer associated with the piece played at `file_path`

   params:
      `file_path`: absolute path to midi file
      `csv_path`: absolute path to csv metadata file
   """
   year = os.path.basename(os.path.dirname(file_path))
   filename_key = os.path.join(year, os.path.basename(file_path))
   filtered_row = [row for row in csv_metadata if row['midi_filename'] == filename_key][0]
   return filtered_row['canonical_composer']
   
def process_midi_file(file_path: str, inp_split: float, max_seqlen: int):
   """
   reads the midi file at the given path and returns a tuple with elements:
      * `inp`: input midi sequence with `inp_split` percent of the original midi sequence
      * `tar`: target midi sequence with `1 - inp_split` percent of the original midi sequence

   """
   midi_track = mido.MidiFile(file_path).tracks[1]
   inp_split = int(len(midi_track) * inp_split)
   inp, tar = np.array_split(midi_track.tracks[1], indices_or_sections=[inp_split])
   # pad

# max_seqlen over entire sequence (inp + tar)
def build_dataset(data_dir, csv_path, composer_path, inp_split, max_seqlen):
   """
   returns a tuple of tensors with shapes:
      * `(num_batches, batch_size, max_inp_seqlen)`: input sequence
      * `(num_batches, batch_size, max_tar_seqlen)`: target sequence
      * `(num_batches, batch_size, num_features)`: sequence features

   `inp_split` and `max_seqlen` are taken from config. features are one hot encoded

   params:
      `data_dir`: abs path to data directory
      `inp_split`: percent of original sequence to take as input. percent of `max_seqlen` to pad 
         input sequences to.
      `max_seqlen`: the max overall sequence length
   """
   with open(csv_path, encoding='utf-8') as midi_metadata:
      csv_metadata = clean_csv(csv.DictReader(midi_metadata))
   with open(composer_path, encoding='utf-8') as composer_metadata:
      composer_data = json.load(composer_metadata)
   for midi_path in os.scandir(data_dir):
      absolute_midi_path = os.path.abspath(midi_path.path)
      inp, tar = process_midi_file(absolute_midi_path, inp_split, max_seqlen)
      composer = get_composer(absolute_midi_path, csv_metadata)
      epoch = get_epoch(composer, composer_data)

def clean_csv(csv_metadata):
   """
   cleans csv metadata:
      * entries with multiple composers are transformed into the most common composer

   """
   return csv_metadata
