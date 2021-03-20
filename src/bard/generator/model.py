import tensorflow as tf
from tensorflow import keras
from ..layers import *

class MusicTransformer(keras.Model):
   def __init__(self, vocab_size, embed_dim, layers, heads, 
               key_dim, value_dim, ffnn_dim, max_relative_pos, dropout_rate, 
               kernel_constraint=None, **kwargs):
      super().__init__(**kwargs)
      self.vocab_size = vocab_size
      self.embed_dim = embed_dim
      self.layers = layers
      self.heads = heads
      self.key_dim = key_dim
      self.value_dim = value_dim
      self.ffnn_dim = ffnn_dim
      self.max_relative_pos = max_relative_pos
      self.dropout_rate = dropout_rate
      self.kernel_constraint = keras.constraints.get(kernel_constraint)
      self.encoders = encoder.EncoderStack(
         units=layers, 
         heads=heads, 
         ffnn_dim=ffnn_dim, 
         max_relative_pos=max_relative_pos,
         dropout_rate=dropout_rate,
         kernel_constraint=kernel_constraint,
         name=f"Encoder Stack ({layers} layers)")
      self.decoders = decoder.DecoderStack(
         units=layers,
         heads=heads,
         ffnn_dim=ffnn_dim,
         max_relative_pos=max_relative_pos,
         dropout_rate=dropout_rate,
         kernel_constraint=kernel_constraint,
         name=f"Decoder Stack ({layers} layers)"
      )
      self.embedding = embedding.Embedding(
         output_dim=embed_dim,
         vocab_size=vocab_size,
         dropout_rate=dropout_rate,
         embeddings_constraint=kernel_constraint,
         name="Encoder Embedding"
      ) # dec_embedding can be shared due to same vocab, but on separate vocabs, use different
      self.linear = keras.layers.Dense(vocab_size, activation=keras.activations.softmax, name="Dense Layer + Softmax")

   def call(self, inputs, targets, features, padding_mask, lookahead_mask, training):
      """
      params:
         inputs: tensor of shape (batch, seqlen)
         targets: tensor of shape (batch, target_seqlen)
         padding_mask: tensor with shape equal to or broadcastable to (batch, target_seqlen, seqlen)
         lookahead_mask: tensor with shape equal to or broadcastable to (batch, target_seqlen, seqlen)
         training: boolean representing whether this forward pass is part of training
         features: tensor of shape (batch, dim) for additional features
      returns:
         result: tensor of shape (batch, target_seqlen, vocab_size)
         enc_weights: encoder weights of shape (batch, )

      """
      enc_out = self.embedding(inputs=inputs)
      enc_out, enc_weights = self.encoders(
         inputs=enc_out, 
         padding_mask=padding_mask, 
         training=training)

      dec_out = self.embedding(inputs=targets)
      dec_out, dec_weights, encdec_weights = self.decoders(
         inputs=dec_out, 
         enc_kv=enc_out, 
         padding_mask=padding_mask, # to mask encoder output in cross attn block where padding exists
         lookahead_mask=lookahead_mask, 
         training=training)

      if features is None:
         features = tf.constant([])
      else:
         features = features[:, tf.newaxis, :] # (batch, 1, dim)
         features = tf.tile(features, [1, dec_out.shape[1], 1]) # (batch, dec_out.shape[1], dim)
      concatenated = keras.layers.concatenate([dec_out, features])

      result = self.linear(concatenated)
      return result, enc_weights, dec_weights, encdec_weights

   def _process(self, data):
      inputs, target = data # ([batch, seqlen], [batch, seqlen + 2])

      # ['start', ...], [..., 'end']
      target_input, target_output = target[:, :-1], target[:, 1:]

      # dec_mask (used in self attn block) combines lookahead_mask with dec_key_mask, 
      # while pad_mask (used in cross attn block) masks enc-kv padding
      pad_mask = masking.create_padding_mask(inputs) # (batch, seqlen)
      dec_mask = masking.create_decoder_mask(target_input) # (batch, q_seqlen, seqlen) q_seqlen == seqlen
      
      # mask key portions, the innermost dimension
      # q * k^T has shape (batch, heads, q_seqlen, seqlen), so mask seqlen
      # shape of mask should be (batch, q_seqlen, seqlen) masking seqlen
      pad_mask = pad_mask[:, tf.newaxis, :] # (batch, q_seqlen=1 [broadcast], seqlen)
      return inputs, target_input, target_output, pad_mask, dec_mask
   
   def train_step(self, data):
      inputs, target_input, target_output, pad_mask, dec_mask = self._process(data)
      with tf.GradientTape() as tape:
         y_pred, *weights = self(
               inputs=inputs, 
               target=target_input,
               # features=??, 
               padding_mask=pad_mask, 
               lookahead_mask=dec_mask, 
               training=True) # (batch, target_seqlen, vocab_size)
         # expect sparse categorical cross entropy
         loss = self.compiled_loss(target_output, y_pred)
      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      self.compiled_metrics.update_state(target_output, y_pred)
      return { metric.name: metric.result() for metric in self.metrics }

   def test_step(self, data):
      inputs, target_input, target_output, pad_mask, dec_mask = self._process(data)
      y_pred, *weights = self(
         inputs=inputs,
         target=target_input,
         # features=??
         padding_mask=pad_mask,
         lookahead_mask=dec_mask,
         training=False
      )
      self.compiled_loss(target_output, y_pred)
      self.compiled_metrics.update_state(target_output, y_pred)
      return { metric.name: metric.result() for metric in self.metrics}

   def get_config(self):
      config = super().get_config()
      config.update({
         'vocab_size': self.vocab_size,
         'embed_dim': self.embed_dim,
         'layers': self.layers,
         'heads': self.heads,
         'key_dim': self.key_dim,
         'value_dim': self.value_dim,
         'ffnn_dim': self.ffnn_dim,
         'max_relative_pos': self.max_relative_pos,
         'dropout_rate': self.dropout_rate,
         'kernel_constraint': keras.constraints.serialize(self.kernel_constraint)
      })
      return config

   @classmethod
   def from_config(cls, config):
      return cls(**config)
