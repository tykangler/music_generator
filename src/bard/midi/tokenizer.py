import tensorflow as tf
from collections import Counter

class MidiTokenizer:
   def __init__(self, vocabulary, sort=True):
      if sort:
         counter = Counter(vocabulary) 
         self.vocabulary = [token for token, _ in counter.most_common()] 
      else:
         self.vocabulary = vocabulary
      # '<pad>': 0
      # '<start>': 1
      # '<end>': 2
      integer_ids = range(3, len(self.vocabulary) + 3)
      self.id_lookup = self._initialize_lookup(self.vocabulary, integer_ids)
      self.vocab_lookup = self._initialize_lookup(integer_ids, self.vocabulary)

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

   def save(self, path):
      with open(path, "w+") as save_file:
         save_file.write("\n".join(self.vocabulary))

   @classmethod
   def load(cls, path):
      with open(path) as load_file:
         vocabulary = load_file.read().split("\n")
      return cls(vocabulary, sort=False)
