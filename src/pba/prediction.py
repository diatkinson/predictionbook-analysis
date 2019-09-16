from __future__ import annotations

import copy
import random
from datetime import datetime
from typing import Any, Dict, List, Union

import dateutil.parser
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser"])


class Prediction:
    def __init__(self,
                 number: int,
                 event: str,
                 outcome: str,
                 user: str,
                 time_created: Union[str, datetime],
                 time_known: Union[str, datetime],
                 responses: List[Response]):
        self.number = number
        self.event = event
        self.event_doc = nlp(event)
        self.outcome = outcome
        self.user = user
        if isinstance(time_created, str):
            self.time_created = dateutil.parser.parse(time_created)
        else:
            self.time_created = time_created
        if isinstance(time_known, str):
            self.time_known = dateutil.parser.parse(time_known)
        else:
            self.time_known = time_known
        self.responses = responses
        self.number = number

    def __eq__(self, other):
        return (self.event == other.event and
                self.user == other.user and
                self.outcome == other.outcome and
                self.time_created == other.time_created and
                self.time_known == other.time_known and
                self.responses == other.responses and
                self.number == other.number)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return self.number

    def show(self):
        response_str = ""
        for resp in self.responses:
            response_str += resp.show()
        return ("----\nEvent: " + self.event +
                "\nOutcome: " + self.outcome +
                "\nUser: " + self.user +
                "\n#: " + (str(self.number) if self.number else "None") +
                "\nTime created: " + str(self.time_created) +
                "\nTime known: " + str(self.time_known) +
                "\n" + response_str + "\n----\n")

    def to_dict(self):
        return {
            'number': self.number,
            'event': self.event,
            'outcome': self.outcome,
            'user': self.user,
            'time_created': str(self.time_created),
            'time_known': str(self.time_known),
            'responses': [r.to_dict() for r in self.responses]
        }

    def right(self) -> bool:
        return self.outcome == "right"

    def wrong(self) -> bool:
        return self.outcome == "wrong"

    def unknown(self) -> bool:
        return self.outcome == "unknown"

    def known(self) -> bool:
        return self.outcome in ["right", "wrong"]

    def withdrawn(self) -> bool:
        return self.outcome == "withdrawn"

    def get_actions(self, act_type) -> List[Any]:
        acts = []
        for resp in self.responses:
            act = resp.actions.get(act_type, None)
            if act is not None:
                acts.append(act)
        return acts

    def credences(self) -> List[int]:
        return self.get_actions("credence")

    def comments(self) -> List[str]:
        return self.get_actions("comment")

    def __repr__(self) -> str:
        return (f"Prediction(event='{self.event}', number={self.number}, outcome='{self.outcome}', "
                f"user='{self.user}', time_created='{self.time_created}', "
                f"time_known='{self.time_known}', responses={self.responses})")

    @staticmethod
    def from_dict(pred: Dict[str, Any]) -> Prediction:
        return Prediction(event=pred['event'],
                          user=pred['user'],
                          outcome=pred['outcome'],
                          number=pred['number'],
                          time_created=pred['time_created'],
                          time_known=pred['time_known'],
                          responses=[Response.from_dict(resp) for resp in pred['responses']])


class Response:
    def __init__(self,
                 user: str,
                 time: Union[str, datetime],
                 actions: Dict[str, Any]):
        self.user = user
        if isinstance(time, str):
            self.time = dateutil.parser.parse(time)
        else:
            self.time = time
        self.actions = actions

    def __eq__(self, other):
        return (self.user == other.user and
                self.time == other.time and
                self.actions == other.actions)

    def __ne__(self, other):
        return not self == other

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user': self.user,
            'time': str(self.time),
            'actions': self.actions}

    def show(self) -> str:
        actions = ""
        for (kind, act) in self.actions.items():
            actions += f"\n    * {kind}: {act}"
        return f"\n  * {self.user} ({self.time}) " + actions

    def __repr__(self) -> str:
        return f"Response(user='{self.user}', time='{self.time}', actions={self.actions})"

    @staticmethod
    def from_dict(resp: Dict[str, Any]) -> Response:
        user = resp['user']
        time = resp['time']
        actions = resp['actions']
        return Response(user, time, actions)


def drop_some_responses(prediction: Prediction) -> Prediction:
    prediction = copy.copy(prediction)
    num_responses = len(prediction.responses)
    responses_to_keep = random.choice(range(num_responses+1))
    prediction.responses = prediction.responses[:responses_to_keep]
    return prediction
