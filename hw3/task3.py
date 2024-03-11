from world import Data, GroundTruth, Map, read_mat, orientation_to_yaw_pitch_roll

from typing import List

import numpy as np


def estimate_covariances(
    gt: List[GroundTruth],
    positions: List[np.ndarray],
    orientations: List[np.ndarray],
    data: List[Data],
) -> np.ndarray:
    covariances: List[np.ndarray] = []
    count = 0

    # covariances = np.zeros((6, 6))
    for index, position in enumerate(positions):
        # If the drone positions predates the first timestamp of our
        # ground truth, we can't properly interpolate or really
        # calculate a covariance off of it, so we ignore it
        if data[index].timestamp < gt[0].timestamp:
            continue

        count += 1

        # Interpolate the ground truth to the same time as the
        # estimated position
        try:
            interpolated = interpolate_ground_truth(gt, data[index])
        except ValueError:
            continue

        # Calculate the difference between the interpolated ground
        # truth and the estimated position
        position_vector = np.array(
            [
                position[0],
                position[1],
                position[2],
                orientations[index][0],
                orientations[index][1],
                orientations[index][2],
            ]
        ).reshape(6, 1)
        error = interpolated - position_vector

        # Calculate the covariance matrix
        covariance = np.dot(error, error.T)
        # covariances += covariance
        covariances.append(covariance)

    # Now that we have all of the 6x6 covariance matricies, we need
    # to find the average of them all
    average_covariance = (1 / (len(covariances) - 1)) * np.sum(covariances, axis=0)

    return average_covariance


def interpolate_ground_truth(gts: List[GroundTruth], estimation: Data) -> np.ndarray:
    # Interpolate the ground truth to the same times as the estimated
    # positions. Essentially, ground truths are recorded at a higher
    # cadence but not necessarily at the same time as the data point;
    # as such, we need to find the two times that surround our given
    # data point timestep and then find the resulting interpolation
    # between those, weighted based on time delta from each.
    #
    # Returns an numpy array (6,1) of the interpolated state vector

    # Get the current timestep from the estimation
    timestamp = estimation.timestamp

    # Find the first ground truth time that is past that timestamp
    a: GroundTruth = None
    b: GroundTruth = None
    for index, gt in enumerate(gts):
        # Skip the first ground truth
        if index == 0:
            continue
        if gt.timestamp > timestamp:
            a = gts[index - 1]
            b = gts[index]
            break
    if a is None or b is None:
        raise ValueError("No ground truth found for given timestamp")

    # Calculate our deltas to each timestamp. We will use this to form a
    # weight for each ground truth state vector for our averaging
    delta_a = timestamp - a.timestamp
    delta_b = b.timestamp - timestamp
    total_delta = b.timestamp - a.timestamp

    percentage_a = delta_a / total_delta
    percentage_b = delta_b / total_delta

    # Finally we create our new interpolated state vector by finding the
    # weighted average between a and b given our percentages as weights
    vector_a = np.array([a.x, a.y, a.z, a.roll, a.pitch, a.yaw]).reshape(6, 1)
    vector_b = np.array([b.x, b.y, b.z, b.roll, b.pitch, b.yaw]).reshape(6, 1)

    interpolated_state = (1 - percentage_a) * vector_a
    interpolated_state += (1 - percentage_b) * vector_b

    # Create a new ground truth object with the interpolated state vector
    return interpolated_state


if __name__ == "__main__":
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

    covariances: List[np.ndarray] = []

    for dataset in datasets:
        base_data, gt = read_mat(dataset)

        map = Map()

        positions: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        times: List[float] = []
        data: List[Data] = []
        for datum in base_data:
            # Estimate the pose of the camera
            if len(datum.tags) == 0:
                continue
            data.append(datum)
            orientation, position = map.estimate_pose(datum.tags)
            positions.append(position)
            orientations.append(orientation_to_yaw_pitch_roll(orientation))
            times.append(datum.timestamp)

        # Now that we have the positions estimated, we need to
        # calculate the covariance matrix. To do this we need to
        # compare the ground truth (interpolated to the same
        # timestep) and the estimated position
        average_covariance = estimate_covariances(gt, positions, orientations, data)

        covariances.append(average_covariance)

        # Print out the dataset and the resulting covariance matrix
        print("**********")
        print(f"Dataset: {dataset}")
        print(average_covariance)

    # Now that we have all of the covariance matricies, we need to
    # find the average of them all
    average_covariance = (1 / len(covariances)) * np.sum(covariances, axis=0)

    print("**********")
    print("Average Covariance")
    print(average_covariance)
    print("**********")
