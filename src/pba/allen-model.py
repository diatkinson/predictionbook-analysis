# TODO: if there aren't any credences, use the initial encoder hidden state instead of zeros for
# the value encoded_credences

import math
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask

from pba.features import calc_prediction_features
from pba.parse import load_json_to_predictions
from pba.prediction import Prediction

CREDENCE_PAD_VALUE = -1


@DatasetReader.register("pba_dataset_reader")
class PBDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 tokenizer: Optional[Tokenizer] = None):
        super().__init__()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tokenizer = tokenizer or WordTokenizer()

    def prediction_to_instance(self, prediction: Prediction) -> Instance:
        assert prediction.known()
        event_tokens = self.tokenizer.tokenize(prediction.event)
        credences = np.array(prediction.credences())
        credence_times = [resp.time for resp in prediction.responses if "credence" in resp.actions]
        credence_time_diffs = np.array([math.log((time - credence_times[0]).total_seconds() + 1)
                                        for time in credence_times])
        assert len(credences) == len(credence_time_diffs)
        features = np.array(calc_prediction_features(prediction)[3:])

        return Instance({"event": TextField(event_tokens, self.token_indexers),
                         "credences": ArrayField(credences, CREDENCE_PAD_VALUE),
                         "credence_time_diffs": ArrayField(credence_time_diffs, CREDENCE_PAD_VALUE),
                         "features": ArrayField(features),
                         "outcome": LabelField(prediction.outcome)})

    def _read(self, file_path: str) -> Iterator[Instance]:
        predictions = load_json_to_predictions(file_path)
        return (self.prediction_to_instance(prediction) for prediction in predictions)


@Model.register("pba_model")
class PBModel(Model):
    '''
    These make sense, if you had a ton of data:
      - cred. seq.
      - cred. seq. + features
      - cred. avg. + features
      - cred. seq. + features + BERT
      - cred. avg + features + BERT

    We'll restrict ourselves to
      - cred. seq.
      - cred. seq. + features
      - cred. seq. + features + BERT
      - BERT
    '''
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder = None,
                 text_encoder: Seq2VecEncoder = None,
                 credence_encoder: Seq2VecEncoder = None,
                 use_features: bool = False) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.text_encoder = text_encoder
        self.credence_encoder = credence_encoder
        self.use_features = use_features

        text_encoder_dim = text_encoder.get_output_dim() if text_encoder else 0
        credence_encoder_dim = credence_encoder.get_output_dim() if credence_encoder else 0
        feature_dim = 5 if use_features else 0
        in_features = text_encoder_dim + credence_encoder_dim + feature_dim
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=1),
            torch.nn.Sigmoid())

    def forward(self,
                # shape: (batch_size, num_tokens)
                event: Dict[str, torch.Tensor],
                # shape: (batch_size, num_credences)
                credences: torch.Tensor,
                # shape: (batch_size, num_credences)
                credence_time_diffs: torch.Tensor,
                # shape: (batch_size, num_features)
                features: torch.Tensor,
                # shape: (batch_size,)
                outcome: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_tensors = []
        if self.text_encoder:
            # shape: (batch_size, text_encoder_hidden_size)
            encoded_event = self.text_encoder(self.text_field_embedder(event),
                                              get_text_field_mask(event))
            input_tensors.append(encoded_event)
        if self.credence_encoder:
            # Check that there's at least one credence
            if credences.shape[1] > 0:
                # shape: (batch_size, num_credences, 2)
                credence_inputs = torch.stack([credences, credence_time_diffs], dim=-1)
                # shape: (batch_size, num_credences, 2)
                credence_mask = credences != CREDENCE_PAD_VALUE
                # shape: (batch_size, credence_encoder_hidden_size)
                encoded_credences = self.credence_encoder(credence_inputs, credence_mask)
            else:
                # (batch_size, credence_encoder output)
                encoded_credences = torch.zeros((credences.shape[0],
                                                 self.credence_encoder.get_output_dim()))
            input_tensors.append(encoded_credences)

        if self.use_features:
            input_tensors.append(features)

        predictions = self.feedforward(torch.cat(input_tensors, dim=-1))
        output = {}
        if outcome is not None:
            output["loss"] = torch.mean((outcome.float() - predictions.squeeze())**2)
        return output
