from math import pi
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from utils import rmse
from world import Coordinate


def rmse_plot(
    gts: List[np.ndarray],
    estimated_positions: List[np.ndarray],
    filtered_positions: List[np.ndarray],
) -> plt.Figure:
    # Calculate our error
    errors_camera: List[np.array] = []
    errors_pf: List[np.array] = []
    for i in range(len(gts)):
        gt_position = np.array([gts[i][0], gts[i][1], gts[i][2]]).reshape((3, 1))
        filtered_position = filtered_positions[i][0:3]

        error = rmse(gt_position, filtered_position)
        errors_pf.append(error)

        error = rmse(gt_position, estimated_positions[i][0:3])
        errors_camera.append(error)

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()

    axes.set_xlabel("Time")
    axes.set_ylabel("Error")
    axes.dist = 11
    axes.set_title(f"RMSE Error")

    axes.plot(errors_pf, label="Particle Filter")
    axes.plot(errors_camera, label="Camera")

    axes.legend()

    return figure


def rmse_methods_plot(
    gts: List[np.ndarray],
    weighted: List[np.ndarray],
    average: List[np.ndarray],
    highest: List[np.ndarray],
) -> plt.Figure:
    # If the gts are 1 more than the
    # estimates, we need to remove the first
    if len(gts) == len(weighted) + 1:
        gts = gts[1:]
    elif len(gts) != len(weighted):
        raise ValueError("Ground truths and estimates are not the same length")

    # Calculate our error
    errors_weighted: List[np.array] = []
    errors_average: List[np.array] = []
    errors_highest: List[np.array] = []
    for i in range(len(gts)):
        gt_position = np.array([gts[i][0], gts[i][1], gts[i][2]]).reshape((3, 1))

        error = rmse(gt_position, weighted[i][0:3])
        errors_weighted.append(error)

        error = rmse(gt_position, average[i][0:3])
        errors_average.append(error)

        error = rmse(gt_position, highest[i][0:3])
        errors_highest.append(error)

    # Print out the mean of each rmse
    print(f"Weighted Average RMSE: {np.mean(errors_weighted)}")
    print(f"Average RMSE: {np.mean(errors_average)}")
    print(f"Highest Weight RMSE: {np.mean(errors_highest)}")

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()

    axes.set_xlabel("Time")
    axes.set_ylabel("Error")

    axes.dist = 11
    axes.set_title(f"RMSE Error")

    axes.plot(errors_weighted, label="Weighted Average")
    axes.plot(errors_average, label="Average")
    axes.plot(errors_highest, label="Highest Weight")

    axes.legend()

    return figure


def isometric_plot(title: str, *args) -> plt.Figure:
    """
    Plots the isometric plots. For each pair in the args, grab, respectively,
    the label for the dataset, and its datapoints.
    """
    if args is None or len(args) == 0:
        raise "No plotting arguments provided"
    elif len(args) % 2 != 0:
        raise "Arguments must be in pairs of label and data"

    more_than_one = len(args) > 2

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
    # if more_than_one:
    #     axes.legend()

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

    estimate_timestamps = timestamps
    if len(estimate_timestamps) != len(estimates):
        estimate_timestamps = timestamps[1:]

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Yaw")
    axs[0].set_title("Yaw")
    axs[0].set_ylim(-pi / 2, pi / 2)
    axs[0].plot(timestamps, yaw_gt, label="Ground Truth")
    axs[0].plot(estimate_timestamps, yaw_estimated, label="Estimated")
    axs[0].legend()

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Pitch")
    axs[1].set_title("Pitch")
    axs[1].set_ylim(-pi / 2, pi / 2)
    axs[1].plot(timestamps, pitch_gt, label="Ground Truth")
    axs[1].plot(estimate_timestamps, pitch_estimated, label="Estimated")
    axs[1].legend()

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Roll")
    axs[2].set_title("Roll")
    axs[2].set_ylim(-pi / 2, pi / 2)
    axs[2].plot(timestamps, roll_gt, label="Ground Truth")
    axs[2].plot(estimate_timestamps, roll_estimated, label="Estimated")
    axs[2].legend()

    positions_figure, axs = plt.subplots(1, 3, figsize=(20, 10))
    positions_figure.suptitle(
        "Trajectory Comparisons of Ground Truth and Estimated Positions"
    )

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Top-Down")
    axs[0].scatter(x_gt, y_gt, c=z_gt, label="Ground Truth")
    axs[0].scatter(x_estimated, y_estimated, label="Estimated")
    axs[0].legend()

    axs[1].set_xlabel("Y")
    axs[1].set_ylabel("Z")
    axs[1].set_title("Side X View")
    axs[1].scatter(y_gt, z_gt, c=z_gt, label="Ground Truth")
    axs[1].scatter(y_estimated, z_estimated, label="Estimated")
    axs[1].legend()

    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Z")
    axs[2].set_title("Side Y View")
    axs[2].scatter(x_gt, z_gt, c=z_gt, label="Ground Truth")
    axs[2].scatter(x_estimated, z_estimated, label="Estimated")
    axs[2].legend()

    return positions_figure, orientations_figure


def rmse_basic_plot(title: str, *args) -> plt.Figure:
    if args is None or len(args) == 0:
        raise "No plotting arguments provided"
    elif len(args) % 2 != 0:
        raise "Arguments must be in pairs of label and data"

    more_than_one = len(args) > 2

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()

    axes.set_xlabel("Time")
    axes.set_ylabel("Error")

    axes.dist = 11
    axes.set_title(title)

    args = list(args)
    while args:
        label = args.pop(0)
        data = args.pop(0)
        axes.plot(data, label=label)

    if more_than_one:
        axes.legend()

    return figure
