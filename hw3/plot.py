import matplotlib.pyplot as plt

from utils import interpolate_ground_truth, rmse
from world import GroundTruth, Data, Coordinate

from typing import List, Optional, Tuple

import numpy as np

from math import pi


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


def isometric_plot(title: str, *args) -> plt.Figure:
    """
    Plots the isometric plots. For each pair in the args, grab, respectively,
    the label for the dataset, and its datapoints.
    """

    fig = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes(projection="3d")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.dist = 11
    axes.set_title(title)

    args = list(args)

    while args:
        label = args.pop(0)
        data = args.pop(0)
        axes.scatter3D(
            [coord[0] for coord in data],
            [coord[1] for coord in data],
            [coord[2] for coord in data],
            c=[coord[2] for coord in data],
            linewidths=0.5,
            label=label,
        )
    axes.legend()

    return fig


def create_overlay_plots(
    ground_truth: List[np.ndarray],
    estimates: List[np.ndarray],
    timestamps: List[float],
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Return tri-view overlay plots for positions and orientations versus
    incoming estimated states.

    Returns positions, orientations, figures respectively
    """

    estimated_positions: List[np.ndarray] = [estimate[0:3] for estimate in estimates]
    estimated_orientations: List[np.ndarray] = [estimate[3:6] for estimate in estimates]

    gt_coordinates = [Coordinate(x=gti[0], y=gti[1], z=gti[2]) for gti in ground_truth]
    estimated_coordinates = [
        Coordinate(x=position[0], y=position[1], z=position[2])
        for position in estimated_positions
    ]

    x_gt = [coord.x for coord in gt_coordinates]
    y_gt = [coord.y for coord in gt_coordinates]
    z_gt = [coord.z for coord in gt_coordinates]

    x_estimated = [coord.x for coord in estimated_coordinates]
    y_estimated = [coord.y for coord in estimated_coordinates]
    z_estimated = [coord.z for coord in estimated_coordinates]

    yaw_gt = [gti[3] for gti in ground_truth]
    pitch_gt = [gti[4] for gti in ground_truth]
    roll_gt = [gti[5] for gti in ground_truth]

    yaw_estimated = [orientation[2] for orientation in estimated_orientations]
    pitch_estimated = [orientation[1] for orientation in estimated_orientations]
    roll_estimated = [orientation[0] for orientation in estimated_orientations]

    orientations_figure, axs = plt.subplots(1, 3, figsize=(20, 10))
    orientations_figure.suptitle(
        "Orientation Comparisons of Ground Truth and Estimated Positions"
    )

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Yaw")
    axs[0].set_title("Yaw")
    axs[0].set_ylim(-pi / 2, pi / 2)
    axs[0].plot(timestamps, yaw_gt, label="Ground Truth")
    axs[0].plot(timestamps[1:], yaw_estimated, label="Estimated")

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Pitch")
    axs[1].set_title("Pitch")
    axs[1].set_ylim(-pi / 2, pi / 2)
    axs[1].plot(timestamps, pitch_gt, label="Ground Truth")
    axs[1].plot(timestamps[1:], pitch_estimated, label="Estimated")

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Roll")
    axs[2].set_title("Roll")
    axs[2].set_ylim(-pi / 2, pi / 2)
    axs[2].plot(timestamps, roll_gt, label="Ground Truth")
    axs[2].plot(timestamps[1:], roll_estimated, label="Estimated")

    positions_figure, axs = plt.subplots(1, 3, figsize=(20, 10))
    positions_figure.suptitle(
        "Trajectory Comparisons of Ground Truth and Estimated Positions"
    )

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Top-Down")
    axs[0].scatter(x_gt, y_gt, c=z_gt, label="Ground Truth")
    axs[0].scatter(x_estimated, y_estimated, c=z_estimated, label="Estimated")

    axs[1].set_xlabel("Y")
    axs[1].set_ylabel("Z")
    axs[1].set_title("Side X View")
    axs[1].scatter(y_gt, z_gt, c=z_gt, label="Ground Truth")
    axs[1].scatter(y_estimated, z_estimated, c=z_estimated, label="Estimated")

    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Z")
    axs[2].set_title("Side Y View")
    axs[2].scatter(x_gt, z_gt, c=z_gt, label="Ground Truth")
    axs[2].scatter(x_estimated, z_estimated, c=z_estimated, label="Estimated")

    return positions_figure, orientations_figure
