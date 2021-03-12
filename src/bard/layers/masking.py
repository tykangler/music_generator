import tensorflow as tf

def create_padding_mask(inputs, pad_value=0):
    """
    returns tensor with same shape as inputs. The result will have 1's where elements equal pad_value,
    and 0's otherwise.
    params:
        inputs: the input tensor
        pad_value: padding token to compare equality against
    """
    mask = tf.cast(tf.math.equal(inputs, pad_value), tf.float32)
    return mask 

def create_lookahead_mask(dim):
    """
    returns tensor with shape of (dim, dim) where the upper right triangle is populated with 1's and the rest 
    populated with 0. 

    mask[i, j] = 1 if j - i > 0 else 0
    """
    mask  = 1 - tf.linalg.band_part(tf.ones((dim, dim)), -1, 0)
    return mask

def create_decoder_mask(inputs, pad_value=0):
    """
    returns tensor with shape (batch, seqlen, seqlen) given input tensor has shape (batch, seqlen). Combines
    the pad mask and lookahead mask for this batch of input sequences.
    """
    pad_mask = create_padding_mask(inputs, pad_value) # (batch, seqlen)
    lookahead_mask = create_lookahead_mask(inputs.shape[1]) # (seqlen, seqlen)
    pad_mask = pad_mask[:, tf.newaxis, :] # (batch, 1, seqlen)
    return tf.maximum(pad_mask, lookahead_mask)
