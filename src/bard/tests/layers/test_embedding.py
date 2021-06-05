import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from bard.layers import embedding

def test_positional_encoding_values():
   values = embedding.positional_embeddings(max_pos=100, dims=32)
   for pos in range(len(values)):
      for i in range(len(values[pos])):
         if i % 2 == 0:
            assert np.isclose(values[pos, i], np.sin(pos / np.power(10000, 2 * (i // 2) / 32)))
         else:
            assert np.isclose(values[pos, i], np.cos(pos / np.power(10000, 2 * (i // 2) / 32)))

def test_layer_forward_pass_creates_correct_shape():
   layer = embedding.Embedding(vocab_size=4, output_dim=16)
