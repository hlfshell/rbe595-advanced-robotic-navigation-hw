from typing import List

import numpy as np
from world import Data, GroundTruth


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def interpolate_ground_truth(gts: List[GroundTruth], estimation: Data) -> np.ndarray:
    # Interpolate the ground truth to the same times as the estimated
    # positions. Essentially, ground truths are recorded at a higher
    # cadence but not necessarily at the same time as the data point;
    # as such, we need to find the two times that surround our given
    # data point timestep and then find the resulting interpolation
    # between those, weighted based on time delta from each.
    #
    # Returns an numpy array (6,1) of the interpolated state vector

    # Get the current timestep from the estimation
    timestamp = estimation.timestamp

    # Find the first ground truth time that is past that timestamp
    a: GroundTruth = None
    b: GroundTruth = None
    for index, gt in enumerate(gts):
        # Skip the first ground truth
        if index == 0:
            continue
        if gt.timestamp > timestamp:
            a = gts[index - 1]
            b = gts[index]
            break
    if a is None or b is None:
        raise ValueError("No ground truth found for given timestamp")

    # Calculate our deltas to each timestamp. We will use this to form a
    # weight for each ground truth state vector for our averaging
    delta_a = timestamp - a.timestamp
    delta_b = b.timestamp - timestamp
    total_delta = b.timestamp - a.timestamp

    percentage_a = delta_a / total_delta
    percentage_b = delta_b / total_delta

    # Finally we create our new interpolated state vector by finding the
    # weighted average between a and b given our percentages as weights
    vector_a = np.array([a.x, a.y, a.z, a.roll, a.pitch, a.yaw]).reshape(6, 1)
    vector_b = np.array([b.x, b.y, b.z, b.roll, b.pitch, b.yaw]).reshape(6, 1)

    interpolated_state = (1 - percentage_a) * vector_a
    interpolated_state += (1 - percentage_b) * vector_b

    # Create a new ground truth object with the interpolated state vector
    return interpolated_state
