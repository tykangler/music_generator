import tensorflow as tf
from collections import OrderedDict

class MidiTokenizer:
   def __init__(self, vocab, default_decoded):
      """can pass in a list of all possible tokens, or tensors representing
      sequences of data. expects python array"""
      ordered_dict = OrderedDict() # to remove dups
      for val in vocab:
         ordered_dict[val] = None
      self.vocab = tf.constant(list(ordered_dict.keys()))
      # '<pad>': 0
      # '<start>': 1
      # '<end>': 2
      integer_ids = tf.constant(range(3, len(self.vocab) + 3), dtype=tf.int64)
      vocab_id_kv_init = tf.lookup.KeyValueTensorInitializer(self.vocab, integer_ids)
      self.id_lookup = tf.lookup.StaticVocabularyTable(vocab_id_kv_init, num_oov_buckets=1)
      id_vocab_kv_init = tf.lookup.KeyValueTensorInitializer(integer_ids, self.vocab)
      self.vocab_lookup = tf.lookup.StaticHashTable(id_vocab_kv_init, default_value=default_decoded)

   def encode(self, vocab):
      "expectes tensor"
      encoded = self.id_lookup.lookup(vocab)
      return encoded

   def decode(self, ids):
      "expects tensor"
      return self.vocab_lookup.lookup(ids)

   def vocab_size(self):
      return self.vocab_lookup.size()

   def save(self, path):
      with open(path, "w+") as save_file:
         save_file.write("\n".join(self.vocabulary))

   @classmethod
   def load(cls, path):
      with open(path) as load_file:
         vocabulary = load_file.read().split("\n")
      return cls(vocabulary, sort=False)
