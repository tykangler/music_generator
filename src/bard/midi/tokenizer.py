import tensorflow as tf

class MidiTokenizer:
   """
   stores the vocabulary for a given input sequence, and encodes elements of the sequence into integer labels.
   A label of 1 and 2 is given for start and end tokens respectively. The input sequence must be a python list
   consisting of objects that implement __hash__ and __eq__. Label mappings are created lazily.
   """
   START_TOKEN = 'start'
   END_TOKEN = 'end'

   def __init__(self):
      self.tokens_to_labels = {
         self.START_TOKEN: 1,
         self.END_TOKEN: 2
      }
      self.labels_to_tokens = self._rebuild_labels_mappings()

   def get_label(self, element, mapping: dict) -> int:
      if element not in mapping:
         mapping[element] = len(mapping) + 1
      return mapping[element]
   
   def encode(self, sequence: list):
      """
      returns a tf.Tensor with integer labels for each element in the given python list, with 
      start and end tokens prepended and appended respectively. Given a sequence of 
      `[Foo(2), Foo(2), Foo(4), Foo(1)]`, the output will be `[1, 3, 3, 4, 5, 2]`.
      """
      return tf.constant([self.get_label(self.START_TOKEN, self.tokens_to_labels)] + 
                        [self.get_label(el, self.tokens_to_labels) for el in sequence] + 
                        [self.get_label(self.END_TOKEN, self.tokens_to_labels)])

   def decode(self, sequence, has_start=True, has_end=True) -> list:
      """
      Given a tf.Tensor with integer labels, this returns a python list with each label replaced
      with its respective object. For proper decoding, the sequence must have been encoded with 
      the same tokenizer.
      params:
         sequence: a tf.Tensor of integer labels
         has_start: whether the sequence has a start token prepended
         has_end: whether the sequence has a end token appended
      """
      self.labels_to_tokens = self._rebuild_labels_mappings()
      decoded = [self.get_label(el, self.labels_to_tokens) for el in sequence.numpy()]
      decoded = decoded[1:] if has_start else decoded
      decoded = decoded[:-1] if has_end else decoded
      return decoded

   def _rebuild_labels_mappings(self):
      return { value: key for key, value in self.tokens_to_labels.items() }

   def __len__(self):
      return len(self.tokens_to_labels)
   
   def vocab_size(self):
      return len(self)

   def start_token(self):
      return self.tokens_to_labels[self.START_TOKEN]

   def end_token(self):
      return self.tokens_to_labels[self.END_TOKEN]
