import tensorflow as tf
from tensorflow import keras
import collections
from bard.generator.model import MusicTransformer
from bard.train.learning import LearningSchedule
from bard.train.loss import PaddedSparseCategoricalCrossentropy
from bard.train.metric import PaddedSparseTopKCategoricalAccuracy
from bard.midi.tokenizer import MidiTokenizer
from bard.midi import meta
from bard.midi import preprocess
from bard.layers.masking import create_padding_mask, create_decoder_mask

EMBED_DIM = 512
LAYERS = 6
HEADS = 8
KEY_DIM = 512
VALUE_DIM = 512
FFNN_DIM = 256
MAX_RELATIVE_POS = 64 # multiples of 4-note chords
DROPOUT_RATE = 0.2
KERNEL_CONSTRAINT = keras.constraints.MaxNorm(max_value=2, axis=0)

BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7

WARMUP_STEPS = 4000
VOCAB_SIZE = 2920 # 16 waits + 32 volumes * 88 notes * 1 instrument + 88 note_offs

MAX_VELOCITY = 128
MIN_VELOCITY = 1

def load_model():
   pass

def save_model(model):
   pass

def create_model():
   """
   creates the transfomer with initial default hyperparameters
   """
   transformer = MusicTransformer(
      vocab_size=VOCAB_SIZE, 
      embed_dim=EMBED_DIM,
      layers=LAYERS,
      heads=HEADS,
      key_dim=KEY_DIM,
      value_dim=VALUE_DIM,
      ffnn_dim=FFNN_DIM,
      max_relative_pos=MAX_RELATIVE_POS,
      dropout_rate=DROPOUT_RATE,
      kernel_constraint=KERNEL_CONSTRAINT)

   lr_sched = LearningSchedule(PARAMS_MODEL['embed_dim'], WARMUP_STEPS)
   optimizer = keras.optimizers.Adam(learning_rate=lr_sched, **PARAMS_OPT)
   loss = PaddedSparseCategoricalCrossentropy(k=3)
   metrics = PaddedSparseTopKCategoricalAccuracy(k=3)
   transformer.compile(
      optimizer=optimizer, 
      loss=loss, 
      metrics=metrics)
   return transformer

def predict(model, input_seq: list, end_token: int, start_token: int, max_len=10):
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
   tokenizer = MidiTokenizer()
   input_seq = tokenizer.encode(input_seq)[tf.newaxis, ...] # (batch=1, seqlen)
   input_padding_mask = create_padding_mask(input_seq)[:, tf.newaxis, :] # (batch=1, q_seqlen=1 (broadcast), seqlen)
   AttentionWeights = collections.namedtuple('AttentionWeights', ['enc_weights', 'dec_weights', 'encdec_weights'])

   while tf.size(output_seq) != max_len and output_seq[-1, -1] != end_token:
      decoder_mask = create_decoder_mask(output_seq) # (batch=1, seqlen, seqlen)
      # generated_seq is made up of ignored first n - 1 tokens, with the final nth token being the prediction
      generated_seq, *weights = model(
         inputs=input_seq, 
         targets=output_seq,
         # features=??,
         padding_mask=input_padding_mask,
         lookahead_mask=decoder_mask,
         training=False) # (batch=1, q_seqlen, vocab_size)
      last_notes = generated_seq[:, -1:, :] # (batch=1, 1, vocab_size)
      generated_labels = tf.argmax(last_notes, axis=-1) 
      # (batch=1, 1), what if gen label = 0, model should be trained to not gen 0
      output_seq = tf.concat([output_seq, generated_labels], axis=-1)

   return output_seq, AttentionWeights(*weights)

def preprocess_raw_midi(midi, midi_len, beat_resolution, velocity_bins):
   quantized_times = preprocess.quantize(
      midi, midi.ticks_per_beat, beat_resolution, midi_len, key=lambda msg: msg.time)
   binned_velocities = preprocess.bin(
      midi, velocity_bins, MAX_VELOCITY, MIN_VELOCITY, midi_len, key=lambda msg: msg.velocity
   )

if __name__ == '__main__':
   pass

# TODOs
# 
# * [x] Add l2 MaxNorm regularization to layers
#     * ~~Use self.losses~~, weights automatically adjusted as updated
#     * [x] Implement for multihead relative attention
# * [x] Configs for kernel constraints 
# * [ ] Pass features into model
# * [ ] Clean and Checkup
# * [ ] Callback Manager
# * [ ] Directory Manager
# * [ ] Modify `test_step()` to use predict method using seqlen of validation sequence
# * [ ] Document all classes and methods with required shapes
