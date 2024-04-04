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


# Open a csv file to append to
with open("noise_test.csv", "a") as f:
    writer = csv.writer(f)

    files = [
        # "./hw4/data/studentdata0.mat",
        # "./hw4/data/studentdata1.mat",
        # "./hw4/data/studentdata2.mat",
        # "./hw4/data/studentdata3.mat",
        "./hw4/data/studentdata4.mat",
        "./hw4/data/studentdata5.mat",
        "./hw4/data/studentdata6.mat",
        "./hw4/data/studentdata7.mat",
    ]

    full_start = time()

    for file in files:
        # Read data
        base_data, gt = read_mat(file)

        for noise in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]:
            for noise_gyro in [0.1, 0.5, 0.75, 1.0, 1.5]:
                # # Let's skip all instances of the noise_gyro being greater
                # # than the noise
                # if noise_gyro > noise:
                #     continue

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

                particle_filter = ParticleFilter(
                    noise_scale=noise, noise_scale_gyro=noise_gyro
                )

                start = time()
                estimates, particles = particle_filter.run(data)
                print()
                print(f"Time taken for {noise:.2f}: {time() - start:.2f} seconds")

                # Calculate per step error
                errors = []
                for i in range(len(estimates)):
                    estimate = estimates[i]
                    estimate_xyz = estimate[0:3]
                    # ground_truth_xyz = np.array([gt[i].x, gt[i].y, gt[i].z])
                    ground_truth_xyz = np.array(
                        [
                            interpolated_gts[i][0],
                            interpolated_gts[i][1],
                            interpolated_gts[i][2],
                        ]
                    )
                    ground_truth_xyz = ground_truth_xyz.reshape((3, 1))

                    error = rmse(estimate_xyz, ground_truth_xyz)
                    errors.append(error)

                print(
                    [
                        file.split(".")[1].split("/")[-1],
                        f"{noise:.3f}",
                        f"{noise_gyro:.4f}",
                        np.mean(errors),
                        np.std(errors),
                        np.max(errors),
                        np.min(errors),
                    ]
                )
                writer.writerow(
                    [
                        file.split(".")[1].split("/")[-1],
                        f"{noise:.3f}",
                        f"{noise_gyro:.4f}",
                        np.mean(errors),
                        np.std(errors),
                        np.max(errors),
                        np.min(errors),
                    ]
                )

    print("Job complete!")
    print(f"Total time taken: {time() - full_start:.2f} seconds")
    print("Fingers crossed!")
