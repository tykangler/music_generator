from tensorflow import keras

PARAMS_MODEL = {
    'embed_dim': 512,
    'layers': 6,
    'heads': 8,
    'key_dim': 512,
    'value_dim': 512,
    'ffnn_dim': 256,
    'max_relative_pos': 64, # multiples of 4-note chords
    'dropout_rate': 0.2,
    'kernel_constraint': keras.constraints.MaxNorm(max_value=2, axis=0)
}

PARAMS_OPT = {
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7,
}
WARMUP_STEPS = 4000
VOCAB_SIZE = 2920 # 16 waits + 32 volumes * 88 notes * 1 instrument + 88 note_offs
