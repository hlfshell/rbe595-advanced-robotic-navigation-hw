from matplotlib import pyplot as plt
import numpy as np

from typing import List

class BayesFilter():
    """
    This is a generic implementation of a Bayes Filter, with no
    application-specific implementation details. It simply is
    initiated with a set of prior measurements and initial belief
    of probability. Then you may predict / update as steps to
    iteratively improve the belief of whatever you're trying to
    estimate.
    """

    def __init__(
        self,
        measurements: List[float],
        initial_probability: float
    ):
        self.measurements = measurements
        self.initial_probability = initial_probability

    def prediction(self):
        pass

    def update(self):
        pass