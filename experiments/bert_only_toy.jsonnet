local bert_only = import "bert_only.jsonnet";

bert_only + {
  "train_data_path": "data/train-predictions-toy.json",
  "validation_data_path": "data/valid-predictions-toy.json",
}