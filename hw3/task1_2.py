import os
from typing import List

import numpy as np
from plot import create_overlay_plots, isometric_plot
from utils import interpolate_ground_truth
from world import Map, orientation_to_yaw_pitch_roll, read_mat

# Make sure hw3/imgs/task1_2 exists
os.makedirs("./hw3/imgs/task1_2", exist_ok=True)

datasets = [
    "./hw3/data/studentdata0.mat",
    "./hw3/data/studentdata1.mat",
    "./hw3/data/studentdata2.mat",
    "./hw3/data/studentdata3.mat",
    "./hw3/data/studentdata4.mat",
    "./hw3/data/studentdata5.mat",
    "./hw3/data/studentdata6.mat",
    "./hw3/data/studentdata7.mat",
]

for dataset in datasets:

    data, gt = read_mat(dataset)

    dataset_name = dataset.split("/")[-1].split(".")[0]

    map = Map()

    positions: List[np.ndarray] = []
    orientations: List[np.ndarray] = []
    times: List[float] = []
    interpolated_gt: List[np.ndarray] = []
    for datum in data:
        # Estimate the pose of the camera
        if len(datum.tags) == 0:
            continue
        # We do the try here because some datasets have the data
        # recording *before* the ground truth vicon is recording
        # and we want to ignore those points as we can't compare
        # it to anything later
        try:
            interpolated_gt.append(interpolate_ground_truth(gt, datum))
        except:
            continue
        orientation, position = map.estimate_pose(datum.tags)
        positions.append(position)
        # orientations.append(orientation_to_yaw_pitch_roll(orientation))
        orientations.append(orientation)
        times.append(datum.timestamp)

    # Create multiplot and isometric plot
    # create_overlay_plots(gt, positions, orientations, times, dataset_name)
    position_figure, orientation_figure = create_overlay_plots(
        interpolated_gt,
        [
            np.array(
                [
                    position[0],
                    position[1],
                    position[2],
                    orientations[i][0],
                    orientations[i][1],
                    orientations[i][2],
                ]
            )
            for i, position in enumerate(positions)
        ],
        times,
    )
    position_figure.savefig(f"./hw3/imgs/task1_2/{dataset_name}_trajectory_merged.png")
    orientation_figure.savefig(
        f"./hw3/imgs/task1_2/{dataset_name}_orientation_merged.png"
    )
    figure = isometric_plot(
        "Ground Truth Trajectory",
        "Ground Truth",
        [[gt.x, gt.y, gt.z] for gt in gt],
    )
    figure.savefig(f"./hw3/imgs/task1_2/{dataset_name}_ground_truth.png")
    figure = isometric_plot(
        "Estimated Trajectory",
        "Camera Estimate",
        [[position[0], position[1], position[2]] for position in positions],
    )
    figure.savefig(f"./hw3/imgs/task1_2/{dataset_name}_estimated.png")

    figure = isometric_plot(
        "Trajectories",
        "Ground Truth",
        [[gt.x, gt.y, gt.z] for gt in gt],
        "Camera Estimate",
        [[position[0], position[1], position[2]] for position in positions],
    )
    figure.savefig(f"./hw3/imgs/task1_2/{dataset_name}_isometric.png")
