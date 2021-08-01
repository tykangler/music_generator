import numpy as np
import mido

REST = 1
NOTE = 2
CONTROL = 3

# note
# 9n note velocity
# control
# Bn controller value

def encode_msg(type, data):
   """
   compresses type and data into a single integer at least 1 byte large.
   params:
      type: 8 bit integer representing the type of the message.
      data: data of the message as list of bytes.
   """
   encoded_bytes = [type] + data
   # num_data_bytes = len(data)
   # encoded = type << (8 * num_data_bytes)
   # for idx, val in enumerate(data):
   #    encoded += val << (8 * (num_data_bytes - idx - 1))
   encoded = int.from_bytes(encoded_bytes, byteorder='big')
   return encoded

def rest_0():
   return encode_msg(REST, [0])

def get_last_nonzero_byte_pos(val):
   """
   gets the (zero-indexed) position of the last nonzero byte (from lsb to msb).
   Example:
      00000000 00000001 00000101 00111101 -> 2
      00000000 00000011 00000000 00000111 00001111 -> 3
   """
   return int(np.floor(np.log2(val) / 8))

def decode_msg(encoded):
   """
   the inverse operation of `encode_msg()`. Takes in a compressed message, and returns a
   pair `(type, data)`. `type` represents the type of the message, while `data` represents the
   data as a list of bytes.
   params:
      encoded: the compressed message
   """
   num_data_bytes = get_last_nonzero_byte_pos(encoded)
   # type = encoded >> (8 * num_data_bytes)
   # data_selector = 1 << (8 * num_data_bytes) - 1
   # data: int = encoded & data_selector
   type, *data = encoded.to_bytes(length=num_data_bytes + 1, byteorder='big')
   return type, data

relevant_messages = {
   'note_on': lambda data: encode_msg(NOTE, data),
   'note_off': lambda data: encode_msg(NOTE, data),
   'control_change': lambda data: encode_msg(CONTROL, data)
}

def _get_message(msg):
   """
   constructs simple two byte messages (Control and Note). Doesn't
   handle instruments. Handles note offs with nonzero velocity
   """
   VELOCITY_BYTE = -1
   msg_data = msg.bytes()[1:] # [status:channel, data...], no time in data bytes
   if msg.type == 'note_off':
      msg_data[VELOCITY_BYTE] = 0
   msg_obj = relevant_messages.get(msg.type, None)
   return msg_obj(msg_data) if msg_obj is not None else msg_obj

def encode(seq, resolution):
   """
   converts mido.Message objects into vocabulary tokens. The time value for each message is
   factored out into its own rest message. If time exceeds resolution, then additional rests are
   created with time equal to resolution. This can be considered an implementation
   of an interface for vocabulary converters.

   The `time` attribute should be quantized, and in beat form.

   Each token is an integer with the first byte representing a type, and the remaining bytes
   representing data for the type
   """
   accum_beats = 0
   for msg in seq:
      accum_beats += msg.time
      msg_obj = _get_message(msg)
      if msg_obj is not None:
         excess_beats = accum_beats % resolution
         for i in range(accum_beats // resolution):
            yield encode_msg(REST, [resolution])
         if excess_beats != 0:
            yield encode_msg(REST, [excess_beats])
         accum_beats = 0
         yield msg_obj

def decode(seq):
   accum_beats = 0
   for encoded_token in seq:
      type, data = decode_msg(encoded_token)
      if type == REST:
         accum_beats += data[0]
      elif type == NOTE:
         yield mido.Message('note_on', note=data[0], velocity=data[1], time=accum_beats)
      elif type == CONTROL:
         yield mido.Message('control_change', control=data[0], value=data[1], time=accum_beats)
