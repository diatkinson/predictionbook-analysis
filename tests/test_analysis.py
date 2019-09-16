from typing import List, Tuple

import hypothesis.strategies as st
import pba.analysis as an
from hypothesis import given

list_sizes = {"min_size": 1, "max_size": 20}
predictions = st.lists(st.tuples(st.booleans(), st.integers(0, 100)), **list_sizes)
true_prediction = st.tuples(st.just(True), st.just(100))
false_prediction = st.tuples(st.just(False), st.just(0))
perfect_prediction = st.one_of([true_prediction, false_prediction])
perfect_predictions = st.lists(perfect_prediction, **list_sizes)
almost_perfect_prediction = st.one_of([st.tuples(st.just(True), st.just(99)),
                                       st.tuples(st.just(False), st.just(1))])
almost_perfect_predictions = st.builds(lambda lst, x: lst + [x],
                                       perfect_predictions,
                                       almost_perfect_prediction)
num_bins = st.integers(1, 100)


@given(credence_results=predictions, num_bins=num_bins)
def test_calibration_score_is_non_negative(credence_results: List[Tuple[bool, int]],
                                           num_bins: int):
    assert an.calibration_score(credence_results, num_bins) >= 0


@given(credence_results=perfect_predictions, num_bins=num_bins)
def test_calibration_score_perfect(credence_results: List[Tuple[bool, int]],
                                   num_bins: int):
    assert 0.0001 > an.calibration_score(credence_results, num_bins)


@given(credence_results=almost_perfect_predictions, num_bins=num_bins)
def test_calibration_score_delta(credence_results: List[Tuple[bool, int]], num_bins: int):
    assert an.calibration_score(credence_results, num_bins) < 0.011


def test_calibration_score_example():
    example = [(True, 100), (True, 50), (True, 0), (False, 50), (False, 70)]
    assert 0.0001 > abs(0.34 - an.calibration_score(example, num_bins=5))
