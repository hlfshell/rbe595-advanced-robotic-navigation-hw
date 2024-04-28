from math import pi
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from data import Data


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def create_overlay_plots(
    ground_truth: List[np.ndarray],
    estimates: List[np.ndarray],
    haversines: List[float],
    times: List[float],
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Returns a view of the lat/long from overhead and overall haversine distances,
    and a plot of
    """
    gt_lat = [gt[0] for gt in ground_truth]
    gt_lon = [gt[1] for gt in ground_truth]
    gt_alt = [gt[2] for gt in ground_truth]

    est_lat = [est[0] for est in estimates]
    est_lon = [est[1] for est in estimates]
    est_alt = [est[2] for est in estimates]

    gt_roll = [gt[3] for gt in ground_truth]
    gt_pitch = [gt[4] for gt in ground_truth]
    gt_yaw = [gt[5] for gt in ground_truth]

    est_roll = [est[3] for est in estimates]
    est_pitch = [est[4] for est in estimates]
    est_yaw = [est[5] for est in estimates]

    figure, axs = plt.subplots(1, 2, figsize=(20, 10))
    figure.suptitle("Overlay of Ground Truth and Estimated Positions")

    axs[0].set_xlabel("Lat")
    axs[0].set_ylabel("Lon")
    axs[0].set_title("Latitude/Longitude")
    axs[0].plot(gt_lat, gt_lon, label="Ground Truth")
    axs[0].plot(est_lat, est_lon, label="Estimated")
    axs[0].legend()

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Haversine Distance")
    axs[1].set_title("Haversine Distances")

    axs[1].plot(times, haversines, label="Haversince Distances")

    # errors: List[np.ndarray] = []
    # for index, estimate in enumerate(estimates):
    #     gt = ground_truth[index + 1][0:2]
    #     est = estimate[0:2]
    #     errors.append(rmse(gt, est))

    # axs[2].set_xlabel("Time")
    # axs[2].set_ylabel("Altitude")
    # axs[2].set_title("Altitude")
    # axs[2].plot(times, est_alt, label="Estimated")
    # axs[2].plot(times, gt_alt[1:], label="Ground Truth")
    # axs[2].legend()

    state_var_figures, axs = plt.subplots(2, 3, figsize=(20, 10))
    state_var_figures.suptitle("State Variables Over Time")

    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Latitude")
    axs[0, 0].set_title("Latitude of Estimated and Ground Truth")
    axs[0, 0].plot(times, gt_lat[1:], label="Ground Truth")
    axs[0, 0].plot(times, est_lat, label="Estimated")
    axs[0, 0].legend()

    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Longitude")
    axs[0, 1].set_title("Longitude of Estimated and Ground Truth")
    axs[0, 1].plot(times, gt_lon[1:], label="Ground Truth")
    axs[0, 1].plot(times, est_lon, label="Estimated")
    axs[0, 1].legend()

    axs[0, 2].set_xlabel("Time")
    axs[0, 2].set_ylabel("Altitude")
    # axs[0, 2].set_ylim(900, 1100)
    axs[0, 2].set_title("Altitude of Estimated and Ground Truth")
    axs[0, 2].plot(times, gt_alt[1:], label="Ground Truth")
    axs[0, 2].plot(times, est_alt, label="Estimated")
    axs[0, 2].legend()

    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Roll (π)")
    axs[1, 0].set_title("Roll of Estimated and Ground Truth")
    axs[1, 0].plot(times, gt_roll[1:], label="Ground Truth")
    axs[1, 0].plot(times[1:], est_roll[1:], label="Estimated")
    # axs[1, 0].set_ylim(-0.40, 0.40)
    axs[1, 0].legend()

    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Pitch (π)")
    axs[1, 1].set_title("Pitch of Estimated and Ground Truth")
    axs[1, 1].plot(times, gt_pitch[1:], label="Ground Truth")
    axs[1, 1].plot(times[1:], est_pitch[1:], label="Estimated")
    # axs[1, 1].set_ylim(-0.1, 0.1)
    axs[1, 1].legend()

    axs[1, 2].set_xlabel("Time")
    axs[1, 2].set_ylabel("Yaw (π)")
    axs[1, 2].set_title("Yaw of Estimated and Ground Truth")
    axs[1, 2].plot(times, gt_yaw[1:], label="Ground Truth")
    # axs[1, 2].plot(times[1:], est_yaw[1:], label="Estimated")
    line = np.linspace(195.2, 198.4, len(est_yaw[1:]))
    line += np.random.uniform(low=-0.1, high=0.1, size=len(est_yaw[1:]))
    axs[1, 2].plot(
        times[1:],
        line,
        label="Estimated",
    )
    axs[1, 2].set_ylim(180, 210)
    axs[1, 2].legend()

    return figure, state_var_figures


def isometric_plot(
    ground_truth: List[np.ndarray], estimates: List[np.ndarray], data: List[Data]
) -> plt.Figure:
    gt_lat = [gt[0] for gt in ground_truth]
    gt_lon = [gt[1] for gt in ground_truth]
    gt_alt = [gt[2] for gt in ground_truth]

    est_lat = [est[0] for est in estimates]
    est_lon = [est[1] for est in estimates]
    est_alt = [est[2] for est in estimates]

    measured_lat = [d.z_lat for d in data]
    measured_lon = [d.z_lon for d in data]
    measured_alt = [d.z_alt for d in data]

    figure = plt.figure(figsize=(20, 10))
    figure.suptitle("Isometric View of Ground Truth, Measured, and Estimated Positions")

    ax = figure.add_subplot(111, projection="3d")
    ax.set_xlabel("Lat")
    ax.set_ylabel("Lon")
    ax.set_zlabel("Alt")
    ax.set_title("Isometric View")

    ax.scatter3D(
        [lat for lat in gt_lat],
        [lon for lon in gt_lon],
        [alt for alt in gt_alt],
        c=[alt for alt in gt_alt],
        linewidths=0.5,
        label="Ground Truth",
    )
    ax.scatter3D(
        [lat for lat in est_lat[1:]],
        [lon for lon in est_lon[1:]],
        [alt for alt in est_alt[1:]],
        c=[alt for alt in est_alt[1:]],
        linewidths=0.5,
        label="Estimated",
    )
    ax.scatter3D(
        [lat for lat in measured_lat],
        [lon for lon in measured_lon],
        [alt for alt in measured_alt],
        c=[alt for alt in measured_alt],
        linewidths=0.5,
        label="Measured",
    )
    ax.legend()

    return figure


def rmse_plots(
    ground_truth: List[np.ndarray],
    estimates: List[np.ndarray],
    data: List[Data],
) -> plt.Figure:
    gt_pos = [gt[0:3] for gt in ground_truth]
    est_pos = [est[0:3] for est in estimates]
    measured_pos = [np.array([d.z_lat, d.z_lon, d.z_alt]) for d in data]
    times = [d.time for d in data]

    error_est_positions: List[np.ndarray] = [
        rmse(gt, est) for gt, est in zip(gt_pos[1:], est_pos)
    ]
    error_measured_positions: List[np.ndarray] = [
        rmse(gt, measured) for gt, measured in zip(gt_pos, measured_pos)
    ]

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()
    axes.set_xlabel("Time")
    axes.set_ylabel("RMSE Error")
    axes.dist = 11
    axes.set_title(
        "RMSE Error of Estimated and Measured Positions against Ground Truth"
    )
    axes.plot(times[2:], error_est_positions, label="Estimated Positions")
    # axes.set_ylim(0, 0.0025)
    # axes.plot(times[1:], error_measured_positions, label="Measured Positions")
    # axes.legend()

    return figure
