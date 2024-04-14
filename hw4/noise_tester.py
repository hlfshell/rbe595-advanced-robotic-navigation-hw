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

from utils import interpolate_ground_truth, rmse

import os

import csv

from typing import List, Dict

from task1 import ParticleFilter
from time import time


# Open a csv file to append to
with open("noise_test.csv", "a") as f:
    writer = csv.writer(f)

    files = [
        # "./hw4/data/studentdata0.mat",
        # "./hw4/data/studentdata1.mat",
        "./hw4/data/studentdata2.mat",
        "./hw4/data/studentdata3.mat",
        "./hw4/data/studentdata4.mat",
        "./hw4/data/studentdata5.mat",
        "./hw4/data/studentdata6.mat",
        "./hw4/data/studentdata7.mat",
    ]

    full_start = time()

    for file in files:
        # Read data
        base_data, gt = read_mat(file)

        for noise in [105.0]:  # [100.0, 105.0, 110.0, 115.0]:
            for noise_gyro in np.arange(0.1, 1.0, 0.05):  # [0.01, 0.1, 0.25, 0.5]:
                # # Let's skip all instances of the noise_gyro being greater
                # # than the noise
                # if noise_gyro > noise:
                #     continue

                positions: List[np.ndarray] = []
                orientations: List[np.ndarray] = []
                times: List[float] = []
                data: List[Data] = []
                interpolated_gts: List[np.ndarray] = []
                for datum in base_data:
                    if not datum.tags:
                        continue
                    try:
                        interpolated_gts.append(interpolate_ground_truth(gt, datum))
                    except:
                        continue
                    data.append(datum)
                    orientation, position = Map().estimate_pose(datum.tags)
                    positions.append(position)
                    orientations.append(orientation_to_yaw_pitch_roll(orientation))
                    times.append(datum.timestamp)

                # Create our gts to be the interpolated ones
                # interpolated_gts = interpolate_from_data(gt, data, True)

                particle_filter = ParticleFilter(
                    particle_count=2_000, noise_scale=noise, noise_scale_gyro=noise_gyro
                )

                start = time()
                estimates, particles = particle_filter.run(
                    data, estimate_method="highest"
                )
                print()
                print(f"Time taken for {noise:.2f}: {time() - start:.2f} seconds")

                # Calculate per step error
                errors = []
                position_errors = []
                orientation_errors = []
                for i in range(len(estimates)):
                    estimate = estimates[i]
                    estimate_state = estimate[0:6]
                    # ground_truth_xyz = np.array([gt[i].x, gt[i].y, gt[i].z])
                    ground_truth_state = np.array(
                        [
                            interpolated_gts[i][0],
                            interpolated_gts[i][1],
                            interpolated_gts[i][2],
                            interpolated_gts[i][3],
                            interpolated_gts[i][4],
                            interpolated_gts[i][5],
                        ]
                    )
                    # ground_truth_state = ground_truth_state.reshape((3, 1))
                    ground_truth_state = ground_truth_state.reshape((6, 1))

                    error = rmse(estimate_state, ground_truth_state)
                    errors.append(error)

                    position_error = rmse(estimate_state[0:3], ground_truth_state[0:3])
                    orientation_error = rmse(
                        estimate_state[3:6], ground_truth_state[3:6]
                    )
                    position_errors.append(position_error)
                    orientation_errors.append(orientation_error)

                print(
                    [
                        file.split(".")[1].split("/")[-1],
                        f"{noise:.3f}",
                        f"{noise_gyro:.4f}",
                        np.mean(errors),
                        np.std(errors),
                        np.max(errors),
                        np.min(errors),
                        np.mean(position_errors),
                        np.std(position_errors),
                        np.max(position_errors),
                        np.min(position_errors),
                        np.mean(orientation_errors),
                        np.std(orientation_errors),
                        np.max(orientation_errors),
                        np.min(orientation_errors),
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
                        np.mean(position_errors),
                        np.std(position_errors),
                        np.max(position_errors),
                        np.min(position_errors),
                        np.mean(orientation_errors),
                        np.std(orientation_errors),
                        np.max(orientation_errors),
                        np.min(orientation_errors),
                    ]
                )

    print("Job complete!")
    print(f"Total time taken: {time() - full_start:.2f} seconds")
    print("Fingers crossed!")
