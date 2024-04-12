import os
from time import time
from typing import Dict, List

import numpy as np
from pf import ParticleFilter
from plot import create_overlay_plots, isometric_plot
from world import Data, Map, orientation_to_yaw_pitch_roll, read_mat
from utils import interpolate_ground_truth

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
            orientations.append(orientation_to_yaw_pitch_roll(orientation))
            times.append(datum.timestamp)

        particle_filter = ParticleFilter(particle_count=5_000)
        start = time()
        estimates_2k, _ = particle_filter.run(data, estimate_method="highest")
        print()
        print(
            f"Time to run 5k particles for {dataset_name}: {time() - start:.2f} seconds"
        )

        isometric_plot(
            f"Trajectories For {dataset_name}" f" Ground Truth",
            "Ground Truth",
            [[gti.x, gti.y, gti.z] for gti in gt],
            "Particle Filter",
            positions,
        ).savefig(f"./hw4/imgs/task1/{dataset_name}_isometric.png")

        positions_overlay, orientations_overlay = create_overlay_plots(
            interpolated_gts,
            estimates_2k,
            times,
        )
        positions_overlay.savefig(f"./hw4/imgs/task1/{dataset_name}_positions.png")
        orientations_overlay.savefig(
            f"./hw4/imgs/task1/{dataset_name}_orientations.png"
        )
