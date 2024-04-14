import os
from time import time
from typing import Dict, List

import numpy as np
from pf import ParticleFilter
from plot import rmse_methods_plot, rmse_basic_plot
from utils import interpolate_ground_truth, rmse
from world import Data, Map, orientation_to_yaw_pitch_roll, read_mat


def method_tests(datasets: List[str]):
    for dataset in datasets:
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

        particle_filter = ParticleFilter(particle_count=2_000)
        start = time()
        estimates_weighted, _ = particle_filter.run(data, estimate_method="weighted")
        print()
        print(f"Time to run weighted for {dataset_name}: {time() - start:.2f} seconds")

        particle_filter = ParticleFilter(particle_count=2_000)
        start = time()
        estimates_average, _ = particle_filter.run(data, estimate_method="average")
        print()
        print(f"Time to run average for {dataset_name}: {time() - start:.2f} seconds")

        particle_filter = ParticleFilter(particle_count=2_000)
        start = time()
        estimates_highest, _ = particle_filter.run(data, estimate_method="highest")
        print()
        print(f"Time to run highest for {dataset_name}: {time() - start:.2f} seconds")

        # Create rmse plots comparing the results of the three methods
        # against eachother
        rmse_methods_plot(
            interpolated_gts,
            estimates_weighted,
            estimates_average,
            estimates_highest,
        ).savefig(f"./hw4/imgs/task2/{dataset_name}_rmse_methods.png")


def particle_count_tests(datasets: List[str]):
    rmse_totals_by_count: Dict[int, List[List[float]]] = {}

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

        for particle_count in [250, 500, 750, 1_000, 2_000, 3_000, 4_000, 5_000]:
            if particle_count not in rmse_totals_by_count:
                rmse_totals_by_count[particle_count] = []

            particle_filter = ParticleFilter(particle_count=particle_count)
            start = time()
            estimates, _ = particle_filter.run(data, estimate_method="highest")
            print()
            print(
                f"Time to run weighted for {dataset_name} @ {particle_count} particles: {time() - start:.2f} seconds"
            )

            rmse_totals = []
            for i in range(len(interpolated_gts[1:])):
                rmse_totals.append(
                    rmse(
                        np.array(
                            [
                                interpolated_gts[i][0],
                                interpolated_gts[i][1],
                                interpolated_gts[i][2],
                            ]
                        ).reshape((3, 1)),
                        estimates[i][0:3],
                    )
                )

            dataset_results[particle_count] = rmse_totals
            rmse_totals_by_count[particle_count].append(rmse_totals)

            print(
                f"Mean RMSE for {dataset_name} @ {particle_count} particles: {np.mean(rmse_totals)}"
            )

        args = []
        for particles, errors in dataset_results.items():
            args.append(str(particles))
            args.append(errors)

        rmse_basic_plot(
            f"RMSE Error for {dataset_name} by Particle Count", *args
        ).savefig(f"./hw4/imgs/task2/{dataset_name}_rmse_particle_count.png")

    # Now we map the average for each dataset by count
    for particle_count, rmse_totals in rmse_totals_by_count.items():
        all_errors = []
        for errors in rmse_totals:
            all_errors.extend(errors)
        print(
            f"Average overall RMSE for {particle_count} particles: {np.mean(all_errors)}"
        )


if __name__ == "__main__":
    os.makedirs(f"./hw4/imgs/task2", exist_ok=True)

    datasets = [
        # "./hw4/data/studentdata0.mat",
        "./hw4/data/studentdata1.mat",
        "./hw4/data/studentdata2.mat",
        "./hw4/data/studentdata3.mat",
        "./hw4/data/studentdata4.mat",
        "./hw4/data/studentdata5.mat",
        "./hw4/data/studentdata6.mat",
        "./hw4/data/studentdata7.mat",
    ]

    method_tests(datasets)
    particle_count_tests(datasets)
