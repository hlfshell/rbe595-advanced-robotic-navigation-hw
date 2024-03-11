from typing import List

from world import (
    Map,
    read_mat,
    orientation_to_yaw_pitch_roll,
    Coordinate,
    plot_trajectory,
    create_overlay_plots,
)

import numpy as np

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
    for datum in data:
        # Estimate the pose of the camera
        if len(datum.tags) == 0:
            continue
        orientation, position = map.estimate_pose(datum.tags)
        positions.append(position)
        orientations.append(orientation_to_yaw_pitch_roll(orientation))
        times.append(datum.timestamp)

    # Create multiplot and isometric plot
    create_overlay_plots(gt, positions, orientations, times, dataset_name)
    figure = plot_trajectory(
        [Coordinate(x=gti.x, y=gti.y, z=gti.z) for gti in gt], "Ground Truth"
    )
    figure.savefig(f"./hw3/imgs/task1_2/{dataset_name}_ground_truth.png")
    figure = plot_trajectory(
        [
            Coordinate(x=position[0], y=position[1], z=position[2])
            for position in positions
        ],
        "Estimated Trajectory",
    )
    figure.savefig(f"./hw3/imgs/task1_2/{dataset_name}_estimated.png")
