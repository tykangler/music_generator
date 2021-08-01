import collections
import sys
import tensorflow as tf
from tensorflow import keras
import mido
import os
import json

from bard.midi import vocabulary
from bard.midi import tokenizer
from bard.layers.masking import create_decoder_mask, create_padding_mask
from bard import constants

PROJECT_ROOT = os.path.abspath("../..")

# predict --------------

def predict(model, input_seq: list, end_token: int, start_token: int, max_len: int):
   """
   infers a continuation of the input sequence until the given end token is reached, or until the continuation
   has a sequence length of max_len.

   params:
      input_seq: a list with shape (seqlen), with elements representing the preprocessed notes/waits of a midi sequence
      end_token: the designated token to end the sequence with
      start_token: the designated start token
      max_len: maximum length of the continued sequence
   """
   output_seq = tf.constant([start_token])[tf.newaxis, :] # (batch=1, q_seqlen=1)
   input_seq = tf.expand_dims(input_seq, axis=0) # (batch=1, seqlen)
   input_padding_mask = create_padding_mask(input_seq)[:, tf.newaxis, :] # (batch=1, q_seqlen=1 (broadcast), seqlen)
   AttentionWeights = collections.namedtuple('AttentionWeights',
      ['enc_weights', 'dec_weights', 'encdec_weights'])
   while tf.size(output_seq) != max_len and output_seq[-1, -1] != end_token:
      decoder_mask = create_decoder_mask(output_seq) # (batch=1, seqlen, seqlen)
      # generated_seq is made up of ignored first n - 1 tokens, with the final nth token being the prediction
      generated_seq, *weights = model(
         inputs=input_seq,
         targets=output_seq,
         padding_mask=input_padding_mask,
         lookahead_mask=decoder_mask,
         training=False) # (batch=1, q_seqlen, vocab_size)
      last_notes = generated_seq[:, -1:, :] # (batch=1, 1, vocab_size)
      generated_labels = tf.argmax(last_notes, axis=-1)
      # (batch=1, 1), what if gen label = 0, model should be trained to not gen 0
      output_seq = tf.concat([output_seq, generated_labels], axis=-1)
   return output_seq, AttentionWeights(*weights)

def parse_midi(stream):
   for time, msg in stream:
      msg = mido.parse(msg)
      msg.time = time
      yield msg

def run(seq, args=None):
   """
   expects `seq` to be a stream of midi events in form: [(time, [status_byte, data_bytes+])*]
   """
   # load config
   with open(constants.config_path) as config_file:
      config = json.load(config_file)
   model_config = config["model"]
   inference_config = config["inference"]

   # load model
   model_path = os.path.join(
      constants.project_root, model_config['save_path'], model_config['name'])
   model = keras.models.load_model(model_path)

   # transform input seq
   parsed = parse_midi(seq)
   vocab_seq = vocabulary.encode(parsed)
   midi_tokenizer = tokenizer.MidiTokenizer.load(
      os.path.join(constants.project_root, model_config['vocab_path']))
   input_seq = midi_tokenizer.encode(vocab_seq)

   output_seq, attention = predict(
      model=model,
      input_seq=input_seq,
      end_token=constants.end_token,
      start_token=constants.start_token,
      max_len=inference_config['max_len']
   )

   decoded = midi_tokenizer.decode(output_seq)
   return vocabulary.decode(decoded), attention

if __name__ == '__main__':
   run(sys.argv[1:])

# fix filenames, prefix with project root
