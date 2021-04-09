import collections
import tensorflow as tf

from bard.layers.masking import create_decoder_mask, create_padding_mask

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
   input_padding_mask = create_padding_mask(input_seq)[:, tf.newaxis, :] # (batch=1, q_seqlen=1 (broadcast), seqlen)
   AttentionWeights = collections.namedtuple('AttentionWeights', 
      ['enc_weights', 'dec_weights', 'encdec_weights'])
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
