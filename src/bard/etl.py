import os
import glob
import mido
import tensorflow as tf

# the plan is to add additional features to the model, but I don't know how to 
# easily differentiate these from noise due to bad data/unstructured data.

# from collections import Counter
# import pandas as pd

# def split_composer_names(composer_names):
#    for name in composer_names:
#       multi_names = name.split(' / ')
#       yield from multi_names

# def get_occurences(composer_names):
#    composer_counter = Counter(split_composer_names(composer_names))
#    return composer_counter

# def replace_multi_names(multi_names, counter):
#    most_common = lambda a, b, counter: a if counter[a] > counter[b] else b
#    return [most_common(name1, name2, counter) for name1, name2 in multi_names]

def get_split_midi_data(inp_split, max_seqlen):
   pass

def build_dataset(midi_dir, max_seqlen, inp_split):
   """
   builds the midi dataset and returns a numpy array with shape (batch_size, )
   """
   midi_data_folders = glob.glob(f'{midi_dir}/[0-9]*')
