import os
from time import time
from typing import Dict, List

import numpy as np
from pf import ParticleFilter
from plot import rmse_basic_plot
from ukf import UKF
from utils import interpolate_ground_truth, rmse
from world import Data, Map, orientation_to_yaw_pitch_roll, read_mat

if __name__ == "__main__":
    os.makedirs("./hw4/imgs/task3", exist_ok=True)

    datasets = [
        "./hw4/data/studentdata0.mat",
        "./hw4/data/studentdata1.mat",
        "./hw4/data/studentdata2.mat",
        "./hw4/data/studentdata3.mat",
        "./hw4/data/studentdata4.mat",
        "./hw4/data/studentdata5.mat",
        "./hw4/data/studentdata6.mat",
        "./hw4/data/studentdata7.mat",
    ]

    for dataset in datasets:
        dataset_results: Dict[int, List[float]] = {}
        dataset_name = dataset.split("/")[-1].split(".")[0]
        base_data, gt = read_mat(dataset)

        interpolated_gts: List[np.ndarray] = []
        positions: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        times: List[float] = []
        data: List[Data] = []
        # Read our dataset
        for datum in base_data:
            # Ignore no april tag points
            if not datum.tags:
                continue
            # If the data points are prior to the
            # motion capture system, ignore it
            # as we can't compare errors against it
            try:
                interpolated_gts.append(interpolate_ground_truth(gt, datum))
            except:
                continue
            data.append(datum)
            orientation, position = Map().estimate_pose(datum.tags)
            positions.append(position)
            # orientations.append(orientation_to_yaw_pitch_roll(orientation))
            orientations.append(orientation)
            times.append(datum.timestamp)

        # Now run it through UKF, and a 2k and 5k particle filter
        ukf = UKF()
        start = time()
        estimates_ukf = ukf.run(data)
        print(f"Time to run UKF for {dataset_name}: {time() - start:.2f} seconds")

        particle_filter = ParticleFilter(particle_count=2_000)
        start = time()
        estimates_2k, _ = particle_filter.run(data, estimate_method="highest")
        print()
        print(
            f"Time to run 2k particles for {dataset_name}: {time() - start:.2f} seconds"
        )

        particle_filter = ParticleFilter(particle_count=5_000)
        start = time()
        estimates_5k, _ = particle_filter.run(data, estimate_method="highest")
        print()
        print(
            f"Time to run 5k particles for {dataset_name}: {time() - start:.2f} seconds"
        )

        # Now calculate the RMSE for each
        rmse_ukf = []
        rmse_2k = []
        rmse_5k = []

        for i in range(len(interpolated_gts[1:])):
            rmse_ukf.append(
                rmse(
                    np.array(
                        [
                            interpolated_gts[i][0],
                            interpolated_gts[i][1],
                            interpolated_gts[i][2],
                        ]
                    ).reshape((3, 1)),
                    estimates_ukf[i][0:3],
                )
            )

            rmse_2k.append(
                rmse(
                    np.array(
                        [
                            interpolated_gts[i][0],
                            interpolated_gts[i][1],
                            interpolated_gts[i][2],
                        ]
                    ).reshape((3, 1)),
                    estimates_2k[i][0:3],
                )
            )

            rmse_5k.append(
                rmse(
                    np.array(
                        [
                            interpolated_gts[i][0],
                            interpolated_gts[i][1],
                            interpolated_gts[i][2],
                        ]
                    ).reshape((3, 1)),
                    estimates_5k[i][0:3],
                )
            )

        # Print the averages for each
        print(f"{dataset_name} - UKF RMSE: {np.mean(rmse_ukf)}")
        print(f"{dataset_name} - 2k RMSE: {np.mean(rmse_2k)}")
        print(f"{dataset_name} - 5k RMSE: {np.mean(rmse_5k)}")

        # Create the plot
        rmse_basic_plot(
            f"RMSE Errors for {dataset_name}",
            "UKF",
            rmse_ukf,
            "2k Particles",
            rmse_2k,
            "5k Particles",
            rmse_5k,
        ).savefig(f"./hw4/imgs/task3/{dataset_name}_rmse.png")
