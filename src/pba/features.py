import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pba.prediction import Prediction

random.seed(0)

# not the cleanest approach...
POLITICS_TOKENS = {token.lower() for token in
                   ["politics", "political", "obama", "mccain", "palin", "biden", "clinton",
                    "trump", "kavanaugh", "gorsuch", "kagan", "sotomayor", "pence", "tillerson",
                    "pompeo", "mnuchin", "mattis", "shanahan", "barr", "vilsack", "sebelius",
                    "farage", "corbyn", "miliband", "clegg", 'labour', 'tory', 'tories',
                    'conservative', 'ukip', 'brexit', "mueller", "senator", "senate"]}
POLITICS_SUBSTRINGS = {token.lower() for token in
                       ["article 50", "jeff sessions", "ted cruz", "marco rubio", "ben carson",
                        "jeb bush", "rand paul", "chris christie", "mike huckabee", "carly fiorina",
                        "rick santorum", "rick perry", "scott walker", "bobby jindal",
                        "lindsey graham", "george pataki", "george bush", "john edwards",
                        "bill richardson", "chris dodd", "dennis kucinich", "mike huckabee",
                        "mitt romney", "ron paul", "fred thompson", "alan keyes", "rudy giuliani",
                        "michele bachmann", "jon huntsman", "newt gingrich", "mitch mcconnell",
                        "paul ryan", "nancy pelosi", "chuck shumer", "matthew whitaker",
                        "ryan zinke", "wilbur ross", "tom price", "alex azar", "betsy devos",
                        "ben carson", "elaine chao", "rick perry", "john kelly", "Kirstjen Nielsen",
                        "Reince Priebus", "Mick Mulvaney", "Scott Pruitt", "Nikki Haley",
                        "Mike Pompeo", "Gina Haspel", "john kerry", "tim geithner", "jack lew",
                        "bob gates", "robert gates", "leon panetta", "chuck hagel", "ash carter",
                        "eric holder", "stacey abrams", "ken salazor", "arne duncan",
                        "janet napolitano", "rahm emanuel", "peter orszag", "susan rice",
                        "samantha power", "theresa may", "david cameron", "michael gove",
                        "nicola sturgeon", "alex salmond", "tim farron", "boris johnson",
                        "marianne williamson", "elizabeth warren", "eric swalwell",
                        "bernie sanders", "tim ryan", "beto o'rourke", "seth moulton",
                        "amy klobuchar", "jay inslee", "john hickenlooper", "kamala harris",
                        "mike gravel", "kirsten gillibrand", "tulsi gabbard", "john delaney",
                        "bill de blasio", "julian castro", "pete buttigieg", "cory booker",
                        "andrew yang",  "bill weld", "martin o'malley", "lincoln chafee",
                        "jim webb", 'democrat', 'republican', 'lib dem', 'liberal democrats',
                        'election', "represenative", "referendum", "brexit"]}


def wait_length_at_creation(prediction: Prediction) -> int:
    # Possibly this should be time between judging and creation
    # Sometimes, the time known is before the time created (not a parsing error). Not sure why.
    return max(0, (prediction.time_known - prediction.time_created).days)


def time_until_known(prediction: Prediction) -> float:
    last_response_time = max([prediction.time_created] +
                             [resp.time for resp in prediction.responses])
    days_until = max(0, (prediction.time_known - last_response_time).days)
    return math.log(1 + days_until)


def politics(prediction: Prediction) -> bool:
    event_str = prediction.event
    event_tokens = {tok.text.lower() for tok in prediction.event_doc}
    if POLITICS_TOKENS & event_tokens:
        return True
    return any(pol_substr in event_str for pol_substr in POLITICS_SUBSTRINGS)


def personal(prediction: Prediction) -> bool:
    return bool({"i", "me", "we", "us", "our", "my"} &
                {tok.text.lower() for tok in prediction.event_doc})


def money(prediction: Prediction) -> bool:
    return "MONEY" in [ent.label_ for ent in prediction.event_doc.ents]


def negative_formulation(prediction: Prediction) -> bool:
    # If we care, we could make this only count when the negative token is outide parens
    return bool({"n't", "not", "no"} & {tok.text.lower() for tok in prediction.event_doc})


def negative_prediction(prediction: Prediction) -> bool:
    return prediction.credences()[0] < 50


def avg_credence(prediction: Prediction) -> float:
    credences = prediction.credences()
    return np.mean(np.mean(credences) / 100 if credences else 0.5)


def difficulty(prediction: Prediction, brier_scores: Dict[str, float]) -> Optional[float]:
    """
    If a prediction is harder than normal, its credences should tend to be more wrong, on average,
    than the credences given by the same users on different predictions. A positive value indicates
    that error on this prediction is higher than you'd expect.
    """
    assert prediction.known()
    difficulties = []
    for response in prediction.responses:
        if 'credence' in response.actions and response.user in brier_scores:
            error = (response.actions['credence']/100 - prediction.right())**2
            # The Brier score represents the average error made by that user. So, if they mess
            # up *more* here, the appended value will be positive
            difficulties.append(error - brier_scores[response.user])
    if difficulties == []:
        return None
    else:
        return np.mean(difficulties)


def calc_prediction_features(prediction: Prediction) -> Tuple[int, bool, float, bool,
                                                              bool, bool, bool, float]:
    assert prediction.known()
    return (prediction.number,
            prediction.right(),
            avg_credence(prediction),
            politics(prediction),
            personal(prediction),
            money(prediction),
            negative_formulation(prediction),
            time_until_known(prediction))


def calc_features(predictions: List[Prediction]) -> pd.DataFrame:
    col_names = ["number", "outcome", "avg_credence", "politics", "personal", "money",
                 "negative_formulation", "time_until_known"]
    records = [calc_prediction_features(prediction) for prediction in predictions]
    return pd.DataFrame.from_records(records, index='number', columns=col_names)
