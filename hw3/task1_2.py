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


data, gt = read_mat("./hw3/data/studentdata0.mat")

map = Map()
# for tag in map.tags:
#     print(
#         f"{tag}- p1: {tuple(map.tags[tag].bottom_left)}, p2: {tuple(map.tags[tag].bottom_right)}, p3: {tuple(map.tags[tag].top_right)}, p4: {tuple(map.tags[tag].top_left)}"
#     )
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
create_overlay_plots(gt, positions, orientations, times)
figure = plot_trajectory(
    [Coordinate(x=gti.x, y=gti.y, z=gti.z) for gti in gt], "Ground Truth"
)
figure.savefig("./hw3/imgs/ground_truth.png")
figure = plot_trajectory(
    [Coordinate(x=position[0], y=position[1], z=position[2]) for position in positions],
    "Estimated Trajectory",
)
figure.savefig("./hw3/imgs/estimated.png")
