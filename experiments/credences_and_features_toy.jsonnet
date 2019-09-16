local credences_and_features = import "credences_and_features.jsonnet";

credences_and_features + {
  "train_data_path": "data/train-predictions-toy.json",
  "validation_data_path": "data/valid-predictions-toy.json",
}