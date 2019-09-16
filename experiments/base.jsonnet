{
  "train_data_path": "data/train-predictions.json",
  "validation_data_path": "data/valid-predictions.json",
  "model": {
    "type": "pba_model",
  },
  "dataset_reader": "pba_dataset_reader",
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 1000,
    "patience": 20,
    "cuda_device": 0,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8,
  },
}
