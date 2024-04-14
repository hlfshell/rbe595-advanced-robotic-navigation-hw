import os
from time import time
from typing import Dict, List

import numpy as np
from pf import ParticleFilter
from plot import create_overlay_plots, isometric_plot
from world import Data, Map, orientation_to_yaw_pitch_roll, read_mat
from utils import interpolate_ground_truth, rmse

if __name__ == "__main__":
    os.makedirs("./hw4/imgs/task1", exist_ok=True)

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
        tmp: List[np.ndarray] = []
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
            tmp.append(
                np.array(
                    [
                        position[0],
                        position[1],
                        position[2],
                        orientation[0],
                        orientation[1],
                        orientation[2],
                    ]
                )
            )

        particle_filter = ParticleFilter(
            particle_count=10_000, noise_scale=105.0, noise_scale_gyro=0.20
        )
        start = time()
        estimates_10k, _ = particle_filter.run(data, estimate_method="highest")
        print()
        print(
            f"Time to run 10k particles for {dataset_name}: {time() - start:.2f} seconds"
        )

        error = rmse(
            np.array([[gti[0], gti[1], gti[2]] for gti in interpolated_gts[1:]]),
            np.array(
                [[estimate[0], estimate[1], estimate[2]] for estimate in estimates_10k]
            ),
        )
        print(f"RMSE for {dataset_name}: {error:.2f}")
        error = rmse(
            np.array([[gti[3], gti[4], gti[5]] for gti in interpolated_gts[1:]]),
            np.array(
                [[estimate[3], estimate[4], estimate[5]] for estimate in estimates_10k]
            ),
        )
        print(f"RMSE for {dataset_name} orientation: {error:.2f}")

        isometric_plot(
            f"Trajectories For {dataset_name}" f" Ground Truth",
            "Ground Truth",
            [[gti.x, gti.y, gti.z] for gti in gt],
            "Particle Filter",
            [[estimate[0], estimate[1], estimate[2]] for estimate in estimates_10k],
        ).savefig(f"./hw4/imgs/task1/{dataset_name}_isometric.png")

        positions_overlay, orientations_overlay = create_overlay_plots(
            interpolated_gts,
            estimates_10k,
            times,
        )
        positions_overlay.savefig(f"./hw4/imgs/task1/{dataset_name}_positions.png")
        orientations_overlay.savefig(
            f"./hw4/imgs/task1/{dataset_name}_orientations.png"
        )
