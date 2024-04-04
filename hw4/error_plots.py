import numpy as np
import matplotlib.pyplot as plt

from world import (
    Data,
    GroundTruth,
    Map,
    orientation_to_yaw_pitch_roll,
    read_mat,
    interpolate_from_data,
)

import os

import csv

from typing import List, Dict

from task1 import ParticleFilter
from time import time


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def create_error_plot(
    gts: List[np.ndarray],
    estimated_positions: List[Data],
    filtered_positions: np.ndarray,
    filename: str,
):

    # Calculate our error
    errors: List[np.array] = []
    for i in range(len(gts)):
        gt_position = np.array([gts[i][0], gts[i][1], gts[i][2]]).reshape((3, 1))
        filtered_position = filtered_positions[i][0:3]

        error = rmse(gt_position, filtered_position)
        errors.append(error)

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()
    # Set the axes such that it won't change
    # axes.set_xlim(0, 3)
    # axes.set_ylim(0, 3)
    # axes.set_zlim(0, 1.5)

    axes.set_xlabel("Time")
    axes.set_ylabel("Error")
    # axes.set_zlabel("Z")
    axes.dist = 11
    axes.set_title(f"RMSE Error")

    axes.plot(errors)

    figure.savefig(filename)


if __name__ == "__main__":

    files = [
        "./hw4/data/studentdata7.mat",
    ]

    for file in files:

        base_data, gt = read_mat(file)

        positions: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        times: List[float] = []
        data: List[Data] = []

        for datum in base_data:
            if not datum.tags:
                continue
            data.append(datum)
            orientation, position = Map().estimate_pose(datum.tags)
            positions.append(position)
            orientations.append(orientation_to_yaw_pitch_roll(orientation))
            times.append(datum.timestamp)

        # Create our gts to be the interpolated ones
        interpolated_gts = interpolate_from_data(gt, data, True)
        interpolated_gts.pop(0)

        particle_filter = ParticleFilter(
            particle_count=250, noise_scale=60, noise_scale_gyro=0.5
        )
        start = time()
        estimates, _ = particle_filter.run(data)
        print()
        print(f"Time taken: {time() - start:.2f} seconds")

        # Create the plot
        create_error_plot(
            interpolated_gts, data, estimates, f"./hw4/imgs/{"studentdata7"}_errors.png"
        )
