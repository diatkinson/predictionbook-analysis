from __future__ import annotations

import collections as col
from datetime import date, datetime
from typing import Counter, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from pba.prediction import Prediction


def credences_per_user(predictions: List[Prediction]) -> Counter[str]:
    users_freqs = []
    for pred in predictions:
        for resp in pred.responses:
            if "credence" in resp.actions:
                users_freqs.append(resp.user)
    return col.Counter(users_freqs)


def predictions_per_user(predictions: List[Prediction]) -> Counter[str]:
    return Counter(pred.user for pred in predictions)


def actions_per_day(predictions: List[Prediction]) -> DefaultDict[date, int]:
    date_counts: DefaultDict[date, int] = col.defaultdict(int)
    for pred in predictions:
        assert date(2008, 1, 1) < pred.time_created.date() <= datetime.now().date()
        date_counts[pred.time_created.date()] += 1
        for resp in pred.responses:
            assert date(2008, 1, 1) < resp.time.date() <= datetime.now().date()
            date_counts[resp.time.date()] += 1
    return date_counts


def bin_actions_by_period(day_counts: DefaultDict[date, int],
                          period_type: str = "Month") -> List[Tuple[date, int]]:
    assert period_type in ["Month", "Year", "Day"]

    period_counts: DefaultDict[date, int] = col.defaultdict(int)
    for day in day_counts:
        period = date(day.year,
                      day.month if period_type in ["Month", "Day"] else 1,
                      day.day if period_type == "Day" else 1)
        period_counts[period] += day_counts[day]
    return sorted(period_counts.items())


def user_start_dates(predictions: List[Prediction]) -> Dict[str, date]:
    '''
    Returns dict keyed by username, with the values being the date of first credence given.
    '''
    start_dates: DefaultDict[str, date] = col.defaultdict(lambda: date(3000, 1, 1))
    for prediction in predictions:
        for resp in prediction.responses:
            if 'credence' in resp.actions:
                user = resp.user
                start_dates[user] = min(start_dates[user], resp.time.date())
    return start_dates


def resolved_credences(predictions: List[Prediction]) -> Dict[int, List[Tuple[bool, float]]]:
    '''
    Returns dict keyed by days since user's first credence, where the values are lists of
    (resolved true?, credence) tuples.
    '''
    user_started = user_start_dates(predictions)
    credence_results: DefaultDict[int, List[Tuple[bool, float]]] = col.defaultdict(list)
    for prediction in predictions:
        if prediction.known():
            for resp in prediction.responses:
                if 'credence' in resp.actions:
                    user = resp.user
                    days_since = (resp.time.date() - user_started[user]).days
                    credence_result = (prediction.right(), resp.actions['credence'])
                    credence_results[days_since].append(credence_result)
    return credence_results


def calibration_score(credence_results: List[Tuple[bool, int]], num_bins: int = 10) -> float:
    """
    We're given a list of (did the event come true?, credence given) tuples.

    Basically, we place each tuple into one of num_bins bins. Then, for each bin, we calculate
    abs(bins_avg_credence / 100 - bin_pct_true), weighted by the number of credences in each bin.

    (This turns out to essentially be the ECE method of Naeini et al, (2015): "Obtaining Well
    Calibrated Probabilities Using Bayesian Binning".)

    I don't have a great intuition for choice of num_bins, and sort of wonder why we wouldn't go
    for the maximum of 100, by analogy to integration. But haven't actually sat down and thought
    about it.
    """
    assert credence_results
    assert all(0 <= cred <= 100 for (outc, cred) in credence_results)
    bin_counts, bin_edges = np.histogram([cred for (_, cred) in credence_results], bins=num_bins)
    # Bin the results tuples
    binned_results: List[List[Tuple[bool, int]]] = []
    for bin_idx in range(num_bins):
        bin_results = []
        for (outc, cred) in credence_results:
            if bin_edges[bin_idx] <= cred <= bin_edges[bin_idx+1]:
                bin_results.append((outc, cred))
        binned_results.append(bin_results)
    # Calculate calibration
    num_results = len(credence_results)
    cal_error = 0
    print(binned_results)
    for bin_idx in range(num_bins):
        bin_results = binned_results[bin_idx]
        if bin_results:
            outcomes, credences = zip(*binned_results[bin_idx])
            bin_weight = (bin_counts[bin_idx] / num_results)
            cal_error += bin_weight * abs(np.mean(outcomes) - np.mean(credences) / 100)
    return cal_error


def brier_score(user: str, predictions: List[Prediction]) -> Optional[float]:
    records: List[Tuple[int, bool]] = []
    for prediction in predictions:
        if prediction.known():
            outcome = prediction.right()
            for response in prediction.responses:
                if response.user == user and 'credence' in response.actions:
                    records.append((response.actions['credence'], outcome))
    if records:
        return np.mean([(int(outcome) - cred/100)**2 for (cred, outcome) in records])
    else:
        return None


def all_users(predictions: List[Prediction]) -> Set[str]:
    """
    Given a list of predictions, return the set of all users who have taken an action.
    """
    users = set()
    for prediction in predictions:
        users.add(prediction.user)
        for resp in prediction.responses:
            users.add(resp.user)
    return users


def brier_scores_by_user(predictions: List[Prediction]) -> Dict[str, float]:
    """
    Given a list of predictions, calculate the Brier score of each user that has given a credence.
    """
    brier_scores = dict()
    users = all_users(predictions)
    for user in tqdm(users):
        score = brier_score(user, predictions)
        if score:
            brier_scores[user] = score
    return brier_scores
