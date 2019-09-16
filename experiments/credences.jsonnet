local base = import "base.jsonnet";

base + {
  "model" +: {
    "credence_encoder": {
      "type": "lstm",
      "input_size": 2,
      "hidden_size": 10,
      "num_layers": 1,
    },
  },
}

/*
layers, hidden_size:
  - (1, 5): 0.188
  - (1, 10): 0.187
  - (1, 20): 0.187
  - (2, 5): 0.189
  - (2, 10): 0.187
  - (2, 20): 0.187
  - (1, 50): 0.187
*/
