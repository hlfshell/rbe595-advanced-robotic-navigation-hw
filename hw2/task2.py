import matplotlib as plt
import numpy as np

from typing import List, Tuple, Dict, NamedTuple


class KalmanFilter:

    def __init__():
        pass


class DataPoint(NamedTuple):
    """
    DataPoint is a measurement consisting of the timestamp,
    it's measurement, and the system's inputs. Note that for
    this project the measurement may be velocity or position
    depending on which file is read
    """

    t: float
    u1: float
    u2: float
    u3: float
    z1: float
    z2: float
    z3: float


def load_data_from_csv(filepath: str) -> List[DataPoint]:
    with open(filepath, "r") as f:
        lines = f.readlines()
        data = [line.strip().split(",") for line in lines]
        data = [DataPoint(*[float(x) for x in row]) for row in data]


def run_filter(data: List[DataPoint], filter: KalmanFilter) -> List[]:
    """
    Given a specific filter and a a list of data points, run the
    filter step by step over the data points and determine what
    the filter would claim the position of the drone is at each
    point
    """
    pass

if __name__ == "__main__":
    pass
