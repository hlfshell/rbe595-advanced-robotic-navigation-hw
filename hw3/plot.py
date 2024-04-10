import matplotlib.pyplot as plt

from utils import interpolate_ground_truth, rmse
from world import GroundTruth, Data

from typing import List, Tuple

import numpy as np


def plot_rmse_loss(
    ground_truths: List[np.ndarray],
    camera_estimations: List[np.ndarray],
    states: np.ndarray,
    timestamps: List[float],
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Given the ground truths, data, and states calculated, return
    plots for position and orientation, respectively.

    Note that we assume that the ground truth and data points are
    A) filtered for only AprilTag positions
    B) Are interpolated and thus have the same timestamps as
    the state vectors
    """

    gt_pos = [np.array([gt[0], gt[1], gt[2]]) for gt in ground_truths]
    gt_orientations = [np.array([gt[3], gt[4], gt[5]]) for gt in ground_truths]

    camera_pos = [
        np.array([estimate[0], estimate[1], estimate[2]])
        for estimate in camera_estimations
    ]
    camera_orientations = [
        np.array([estimate[3], estimate[4], estimate[5]])
        for estimate in camera_estimations
    ]

    states_pos = [state[0:3] for state in states]
    states_orientations = [state[3:6] for state in states]

    # Calculate all RMSEs
    camera_position_rmse = [
        rmse(gt, camera_pos[index]) for index, gt in enumerate(gt_pos)
    ]
    camera_orientation_rmse = [
        rmse(gt, camera_orientations[index]) for index, gt in enumerate(gt_orientations)
    ]

    # Remember, we skip the first ground truth at the equivalent time because
    # we don't have a previous state to compare it to
    state_position_rmse = [
        rmse(gt, states_pos[index]) for index, gt in enumerate(gt_pos[1:])
    ]
    state_orientation_rmse = [
        rmse(gt, states_orientations[index])
        for index, gt in enumerate(gt_orientations[1:])
    ]

    position_figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()
    axes.set_title("Position RMSE Loss")
    axes.set_xlabel("Time")
    axes.set_ylabel("RMSE Loss (m)")

    # Plot the RMSEs
    axes.plot(timestamps, camera_position_rmse, label="Camera Estimation Error")
    axes.plot(timestamps[1:], state_position_rmse, label="UKF Error")
    axes.legend()

    orientation_figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()
    axes.set_title("Orientation RMSE Loss")
    axes.set_xlabel("Time")
    axes.set_ylabel("RMSE Loss (rad)")

    # Plot the RMSEs
    axes.plot(timestamps, camera_orientation_rmse, label="Camera Estimation Error")
    axes.plot(timestamps[1:], state_orientation_rmse, label="UKF Error")
    axes.legend()

    return position_figure, orientation_figure
