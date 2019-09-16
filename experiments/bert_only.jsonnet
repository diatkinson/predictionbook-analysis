local base = import "base.jsonnet";
local bert_model = "bert-large-cased";

base + {
  "dataset_reader": {
    "type": "pba_dataset_reader",
    "tokenizer": {
      "word_splitter": "bert-basic",
    },
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
        "do_lowercase": true,
      },
    },
  },
  "model" +: {
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
      },
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained",
          "pretrained_model": bert_model,
          "top_layer_only": true,
          "requires_grad": true
        },
      },
    },
    "text_encoder": {
      "type": "bert_pooler",
      "pretrained_model": bert_model,
    },
  },
}
