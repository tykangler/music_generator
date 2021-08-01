from typing import DefaultDict
import tensorflow as tf
import numpy as np
import pytest

from bard.midi import tokenizer, vocabulary

MAX_SIZE = 1000
DEFAULT_DECODED = '0'

@pytest.mark.parametrize("inputs, expected", [
   (["a", "a", "b", "c", "a", "d", "f"], 5),
   (["a", "b", "c", "d", "e", "f"], 6),
])
def test_tokenizer_has_correct_vocab_size_with_sequence(inputs, expected):
   tok = tokenizer.MidiTokenizer(inputs, default_decoded=DEFAULT_DECODED)
   assert bool(tf.reduce_all(tok.vocab_size() == expected))

@pytest.mark.parametrize("inputs, expected", [
   (["a", "a", "b", "c", "a", "d", "f"], tf.constant([3, 3, 4, 5, 3, 6, 7], dtype=tf.int64)),
   (["a", "b", "b", "c", "d", "d", "d", "e"], tf.constant([3, 4, 4, 5, 6, 6, 6, 7], dtype=tf.int64))
])
def test_tokenizer_encodes_correctly_with_sequence(inputs, expected):
   tok = tokenizer.MidiTokenizer(inputs, default_decoded=DEFAULT_DECODED)
   encoded = tok.encode(tf.constant(inputs))
   assert bool(tf.reduce_all(encoded == expected))

@pytest.mark.parametrize("inputs", [
   (["a", "a", "b", "c", "a", "d", "f"]), (["a", "b", "b", "c", "d", "d", "d", "e"])
])
def test_tokenizer_decodes_to_same_sequence(inputs):
   tok = tokenizer.MidiTokenizer(inputs, default_decoded=DEFAULT_DECODED)
   encoded = tok.encode(tf.constant(inputs))
   assert bool(tf.reduce_all(tf.constant(inputs) == tok.decode(encoded)))
