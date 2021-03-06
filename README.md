# music_generator

This project acts as a backend to **Bard**, a music generator web app. 

## Configuration

The default config is the following:

```json
{
   "app": {
      "log_dir": "./logs",
      "model_dir": "./models",
      "data_dir": "./data",
      "checkpoint_dir": "./models/ckpts"
   },
   "midi": {
      "notes": 88,
      "velocity_bins": 32,
      "rest_resolution": 32,
      "instruments": ["piano"],
      "max_seqlen": 300000,
      "inp_split": 0.2
   },
   "model": {
      "initial_params": {
         "model": {
            "embed_dim": 512,
            "layers": 6,
            "key_dim": 512,
            "value_dim": 512,
            "ffnn_dim": 256,
            "max_relative_pos": 64,
            "dropout_rate": 0.2,
            "kernel_constraint": {
               "class_name": "MaxNorm",
               "config": {
                  "max_value": 2,
                  "axis": 0
               }
            }
         }, 
         "learning_schedule": {
            "dim": 512,
            "warmup_steps": 4000
         },
         "optimizer": {
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7
         },
         "loss": {
            "k": 3
         },
         "metric": {
            "k": 3
         }
      }
   }
}
```



## Model Architecture

## Program Structure
