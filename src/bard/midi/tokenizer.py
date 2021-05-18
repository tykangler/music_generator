import tensorflow as tf
from collections import Counter
from itertools import chain

class MidiTokenizer:
   def __init__(self, vocabulary):
      counter = Counter(vocabulary) 
      vocabulary = [token for token, _ in counter.most_common()]
      # '<pad>': 0
      # '<start>': 1
      # '<end>': 2
      integer_ids = range(3, len(vocabulary) + 3)
      self.id_lookup = self._initialize_lookup(vocabulary, integer_ids)
      self.vocab_lookup = self._initialize_lookup(integer_ids, vocabulary)

   def _initialize_lookup(self, keys, values):
      kv_tensor_init = tf.lookup.KeyValueTensorInitializer(
         keys, values, key_dtype=tf.string, value_dtype=tf.int64)
      return tf.lookup.StaticVocabularyTable(kv_tensor_init, num_oov_buckets=1)

   def encode(self, vocab):
      encoded = self.id_lookup.lookup(vocab)
      return encoded

   def decode(self, ids):
      return self.vocab_lookup.lookup(ids)

   def vocab_size(self):
      return self.vocab_lookup.size()
