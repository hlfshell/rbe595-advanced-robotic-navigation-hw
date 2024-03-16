from typing import List, Tuple

import numpy as np
from world import Data, GroundTruth, Map, orientation_to_yaw_pitch_roll, read_mat
from task3 import interpolate_ground_truth


# Covariance matrix below is calculated in task3.py through
# the analyzing the data of the drone's experiments
covariance_matrix = np.array(
    [
        [
            7.09701409e-03,
            2.66809900e-05,
            1.73906943e-03,
            4.49014777e-04,
            3.66195490e-03,
            8.76154421e-04,
        ][
            2.66809900e-05,
            4.70388499e-03,
            -1.33432420e-03,
            -3.46505064e-03,
            1.07454548e-03,
            -1.69184839e-04,
        ][
            1.73906943e-03,
            -1.33432420e-03,
            9.00885499e-03,
            1.80220246e-03,
            3.27846190e-03,
            -1.11786368e-03,
        ][
            4.49014777e-04,
            -3.46505064e-03,
            1.80220246e-03,
            5.27060654e-03,
            1.01361187e-03,
            -5.86487142e-04,
        ][
            3.66195490e-03,
            1.07454548e-03,
            3.27846190e-03,
            1.01361187e-03,
            7.24994152e-03,
            -1.36454993e-03,
        ][
            8.76154421e-04,
            -1.69184839e-04,
            -1.11786368e-03,
            -5.86487142e-04,
            -1.36454993e-03,
            1.21162646e-03,
        ]
    ]
)


def find_sigma_points(mu: np.ndarray, kappa: float) -> np.ndarray:
    pass


def update(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    pass


def predict(mu: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    pass


def rmse(
    gts: List[GroundTruth], estimated_positions: List[Data]
) -> Tuple(np.ndarray, np.ndarray):
    """
    Given a set of ground truth points, return the total RMSE between the groundtruth
    and estimated positions. The return is the error in the position, and then the
    orientation.
    """
    for estimate in estimated_positions:
        gt = interpolate_ground_truth(gts, estimate)
    # TODO
    pass


def ukf(estimated_positions: List[Data]) -> List[np.ndarray]:
    """
    Given a set of estimated positions, return the estimated positions after running
    the Unscented Kalman Filter over the data
    """
