from __future__ import annotations
from cv2 import solvePnP, Rodrigues
import numpy as np

import matplotlib.pyplot as plt

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from scipy.io import loadmat

import cv2


class ImageCoordinate(NamedTuple):
    x: int
    y: int

    def to_coordinates() -> Coordinate:
        pass


class Coordinate(NamedTuple):
    x: float
    y: float
    z: float


class AprilTag(NamedTuple):
    """
    AprilTags are named tuples with the an id field,
    followed by fields p1 through p4, wherein they
    represent the coordinate locations of each corner
    of the april tag in the following order:
    bottom left
    bottom right
    top right
    top left
    """

    id: str
    bottom_left: Union[Coordinate, ImageCoordinate]
    bottom_right: Union[Coordinate, ImageCoordinate]
    top_right: Union[Coordinate, ImageCoordinate]
    top_left: Union[Coordinate, ImageCoordinate]


class Data(NamedTuple):
    """
    Data is the incoming row data from the raw data for this
    project. The data has the following fields:
    img - the raw image data from the drone
    tags - a list of AprilTag objects that were observed
        in the image, if any were present
    timestamp - the time in seconds of the measurement
    rpy - a 3x1 orientation vector of roll pitch and yaw
        measured in radians via the IMU
    drpy - a 3x1 angular velocity vector measured in radians
        per second via the IMU
    acc - a 3x1 acceleration vector measured in m/s^2 via the
        IMU
    """

    img: np.ndarray
    tags: List[AprilTag]
    timestamp: float
    rpy: np.ndarray
    drpy: np.ndarray
    acc: np.ndarray


class GroundTruth(NamedTuple):
    """
    GroundTruth is data read from our motion capture system
    with the following format:
    timestamp - the time in seconds of the measurement
    x, y, z - the position of the drone in meters
    roll, pitch, yaw - the orientation of the drone in radians
    vx, vy, vz - the velocity of the drone in m/s per axis
    wx, wy, wz - the angular velocity of the drone in rad/s per
        axis
    """

    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    vy: float
    vz: float
    wx: float
    wy: float
    wz: float


class Map:
    """
    Map is a class that maintains the world coordinates
    of each AprilTag in the world and provides helper
    functions to transform between an observed AprilTag
    and its world coordinates.
    """

    def __init__(
        self,
        tags: Optional[Dict[int, Coordinate]] = None,
        camera_matrix: Optional[np.ndarray] = None,
        distortion_coefficients: Optional[np.ndarray] = None,
    ):
        if tags is None:
            self.initialize_tags()
        else:
            self.tags = tags

        if camera_matrix is None:
            self.camera_matrix = np.array(
                [[314.1779, 0, 199.4848], [0, 314.2218, 113.7838], [0, 0, 1]]
            )
        else:
            self.camera_matrix = camera_matrix

        if distortion_coefficients is None:
            self.distortion_coefficients = np.array(
                [-0.438607, 0.248625, 0.00072, -0.000476, -0.0911]
            )
        else:
            self.distortion_coefficients = distortion_coefficients

    def initialize_tags(self):
        """
        This function does hard-coded calculation of initial tag
        positioning based on a set of information provided within
        the assignment.

            Each tag in the map has a unique ID that can be found in
            the file parameters.txt. The tags are arranged in a
            12 x 9 grid. The top left corner of the top left tag as
            shown in the map image below should be used as coordinate
            (0, 0) with the x coordinate going down the mat and the y
            coordinate going to the right. The z coordinate for all
            corners is 0. Each tag is a 0.152 m square with 0.152 m
            between tags, except for the space between columns 3 and 4,
            and 6 and 7, which is 0.178 m. Using this information, you
            can compute the location of every corner of every tag in
            the world frame.
        """
        tags: Dict[int, AprilTag] = {}
        tag_map = [
            [0, 12, 24, 36, 48, 60, 72, 84, 96],
            [1, 13, 25, 37, 49, 61, 73, 85, 97],
            [2, 14, 26, 38, 50, 62, 74, 86, 98],
            [3, 15, 27, 39, 51, 63, 75, 87, 99],
            [4, 16, 28, 40, 52, 64, 76, 88, 100],
            [5, 17, 29, 41, 53, 65, 77, 89, 101],
            [6, 18, 30, 42, 54, 66, 78, 90, 102],
            [7, 19, 31, 43, 55, 67, 79, 91, 103],
            [8, 20, 32, 44, 56, 68, 80, 92, 104],
            [9, 21, 33, 45, 57, 69, 81, 93, 105],
            [10, 22, 34, 46, 58, 70, 82, 94, 106],
            [11, 23, 35, 47, 59, 71, 83, 95, 107],
        ]

        # We now need to calculate the p1 through p4 values of each
        # AprilTag
        extra_offset = 0.178 - 0.152
        for row_index, row in enumerate(tag_map):
            x_offset = 0.152 * row_index * 2
            for tag_index, tag in enumerate(row):
                y_offset = 0.152 * tag_index * 2
                if row_index + 1 >= 3:
                    y_offset += extra_offset
                if row_index + 1 >= 6:
                    y_offset += extra_offset

                top_left = Coordinate(x_offset, y_offset, 0)
                top_right = Coordinate(x_offset, y_offset + 0.152, 0)
                bottom_right = Coordinate(x_offset + 0.152, y_offset + 0.152, 0)
                bottom_left = Coordinate(x_offset + 0.152, y_offset, 0)

                tags[tag] = AprilTag(
                    id=tag,
                    top_left=top_left,
                    top_right=top_right,
                    bottom_right=bottom_right,
                    bottom_left=bottom_left,
                )
        self.tags = tags

    def estimate_pose(self, tags: List[AprilTag]) -> Tuple[np.ndarray, np.ndarray]:
        """
        estimate_pose will, given a list of observed AprilTags,
        pair them with their real world coordinates in order to
        estimate the orientation and position of the camera at
        that moment in time.
        """
        world_points = []
        image_points = []
        for tag in tags:
            world_points.append(self.tags[tag.id].bottom_left)
            world_points.append(self.tags[tag.id].bottom_right)
            world_points.append(self.tags[tag.id].top_right)
            world_points.append(self.tags[tag.id].top_left)

            image_points.append(tag.bottom_left)
            image_points.append(tag.bottom_right)
            image_points.append(tag.top_right)
            image_points.append(tag.top_left)
        world_points = np.array(world_points)
        image_points = np.array(image_points)

        _, orientation, position = solvePnP(
            world_points,
            image_points,
            self.camera_matrix,
            self.distortion_coefficients,
            flags=0,
        )

        # Build our kinematic transform frame from the camera offset provided
        # XYZ = [-0.04, 0.0, -0.03];
        # Yaw = pi/4;
        frame_transform = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, -0.04],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0, 0],
                [0, 0, 1, -0.03],
                [0, 0, 0, 1],
            ]
        )

        # Convert the orientation to rotation matrix via
        # Rodrigues' formulae
        orientation = Rodrigues(orientation)[0]

        current_camera_position = np.array(
            [
                np.concatenate((orientation[0:3, 0], -position[0])),
                np.concatenate((orientation[0:3, 1], position[1])),
                np.concatenate((orientation[0:3, 2], position[2])),
                [0, 0, 0, 1],
            ]
        )

        # Now we can translate the position back to the center of
        # the drone
        # drone_state = np.dot(current_camera_position, np.linalg.inv(frame_transform))
        drone_state = np.dot(np.linalg.inv(frame_transform), current_camera_position)
        # drone_state = np.dot(frame_transform, current_camera_position)

        position = drone_state[0:3, 3]
        # Convert the rotation matrix back to a vector
        orientation = Rodrigues(drone_state[0:3, 0:3])[0]

        return orientation, position


def read_mat(filepath: str) -> Tuple[List[Data], List[GroundTruth]]:
    """
    Read the .mat file from the given filepath and return the
    data and ground truth as a tuple of lists of Data and
    GroundTruth objects, respectively.
    """

    mat = loadmat(filepath, simplify_cells=True)

    data_mat = mat["data"]
    time_mat = mat["time"]
    vicon_mat = mat["vicon"]

    # Build our data list
    data: List[Data] = []
    for index, datum in enumerate(data_mat):
        tags: List[AprilTag] = []

        # Sometimes the datum["id"] is an int when it's
        # only a single item instead of the expected list;
        # so handle that case
        # MATLAB unfortunately saves items as a scalar (losing)
        # an order of dimensionality when a single AprilTag is
        # present; meaning we have to do type checks to see if
        # we encountered a scenario. Luckily if one check notices
        # it, we can safely assume the others are similarly
        # affected.
        if isinstance(datum["id"], int):
            datum["id"] = [datum["id"]]
            # p1 through p4 are in the format of [[x], [y]]
            # so we have to convert [x, y] to that higher
            # dimensionality.
            for point in ["p1", "p2", "p3", "p4"]:
                datum[point] = [[datum[point][0]], [datum[point][1]]]

        for index, id in enumerate(datum["id"]):
            tags.append(
                AprilTag(
                    id=id,
                    bottom_left=ImageCoordinate(
                        datum["p1"][0][index], datum["p1"][1][index]
                    ),
                    bottom_right=ImageCoordinate(
                        datum["p2"][0][index], datum["p2"][1][index]
                    ),
                    top_right=ImageCoordinate(
                        datum["p3"][0][index], datum["p3"][1][index]
                    ),
                    top_left=ImageCoordinate(
                        datum["p4"][0][index], datum["p4"][1][index]
                    ),
                )
            )

        data.append(
            Data(
                img=datum["img"],
                tags=tags,
                timestamp=datum["t"],
                rpy=datum["rpy"],
                drpy=datum["drpy"],
                acc=datum["acc"],
            )
        )

    # Build our ground truth list
    ground_truth: List[GroundTruth] = []
    for index, moment in enumerate(time_mat):
        ground_truth.append(
            GroundTruth(
                timestamp=moment,
                x=vicon_mat[0][index],
                y=vicon_mat[1][index],
                z=vicon_mat[2][index],
                roll=vicon_mat[3][index],
                pitch=vicon_mat[4][index],
                yaw=vicon_mat[5][index],
                vx=vicon_mat[6][index],
                vy=vicon_mat[7][index],
                vz=vicon_mat[8][index],
                wx=vicon_mat[9][index],
                wy=vicon_mat[10][index],
                wz=vicon_mat[11][index],
            )
        )

    return data, ground_truth


def plot_trajectory(
    trajectory: List[Coordinate],
    title: str = "",
    view: Optional[str] = None,
):
    """
    Given a list of coordinates as a trajectory plot the
    trajectory in 3D space.
    """

    x = [coord.x for coord in trajectory]
    y = [coord.y for coord in trajectory]
    z = [coord.z for coord in trajectory]

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes(projection="3d")
    if view == "top" or view == "z":
        axes.view_init(elev=90.0, azim=0)
    elif view == "x":
        axes.view_init(elev=0.0, azim=90.0)
    elif view == "y":
        axes.view_init(elev=0.0, azim=180.0)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.dist = 11
    axes.set_title(title)

    axes.scatter3D(x, y, z, c=z, linewidths=0.5)

    return figure


def create_trajectory_plots(
    ground_truth: List[GroundTruth],
    estimated_positions: List[np.ndarray],
):
    gt_coordinates = [Coordinate(x=gti.x, y=gti.y, z=gti.z) for gti in ground_truth]
    estimated_coordinates = [
        Coordinate(x=position[0], y=position[1], z=position[2])
        for position in estimated_positions
    ]

    x_gt = [coord.x for coord in gt_coordinates]
    y_gt = [coord.y for coord in gt_coordinates]
    z_gt = [coord.z for coord in gt_coordinates]

    x_estimated = [coord.x for coord in estimated_coordinates]
    y_estimated = [coord.y for coord in estimated_coordinates]
    z_estimated = [coord.z for coord in estimated_coordinates]

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Trajectory comparisons of Ground Truth and Estimated Positions")

    # Plot the trajectory of the ground truth and the estimated
    # positions from the top-down view

    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    axs[0, 0].dist = 11
    axs[0, 0].set_title("Top-Down Ground Truth Trajectory")
    axs[0, 0].scatter(x_gt, y_gt, z_gt, c=z_gt, linewidths=0.5)

    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    axs[1, 0].dist = 11
    axs[1, 0].set_title("Top-Down Estimated Trajectory")
    axs[1, 0].scatter(
        x_estimated, y_estimated, z_estimated, c=z_estimated, linewidths=0.5
    )

    axs[0, 1].set_xlabel("y")
    axs[0, 1].set_ylabel("z")
    axs[0, 1].dist = 11
    axs[0, 1].set_title("X Ground Truth Trajectory")
    axs[0, 1].scatter(y_gt, z_gt, c=z_gt, linewidths=0.5)

    axs[1, 1].set_xlabel("y")
    axs[1, 1].set_ylabel("z")
    axs[1, 1].dist = 11
    axs[1, 1].set_title("X Estimated Trajectory")
    axs[1, 1].scatter(y_estimated, z_estimated, c=z_estimated, linewidths=0.5)

    axs[0, 2].set_xlabel("x")
    axs[0, 2].set_ylabel("z")
    axs[0, 2].dist = 11
    axs[0, 2].set_title("Y Ground Truth Trajectory")
    axs[0, 2].scatter(x_gt, z_gt, c=z_gt, linewidths=0.5)

    axs[1, 2].set_xlabel("x")
    axs[1, 2].set_ylabel("z")
    axs[1, 2].dist = 11
    axs[1, 2].set_title("Y Estimated Trajectory")
    axs[1, 2].scatter(x_estimated, z_estimated, c=z_estimated, linewidths=0.5)

    fig.savefig("./hw3/imgs/combined.png")


def create_orientation_plots(
    ground_truth: List[GroundTruth],
    estimated_orientations: List[np.ndarray],
):
    yaw_gt = [gti.yaw for gti in ground_truth]
    pitch_gt = [gti.pitch for gti in ground_truth]
    roll_gt = [gti.roll for gti in ground_truth]

    yaw_estimated = [orientation[2] for orientation in estimated_orientations]
    pitch_estimated = [orientation[1] for orientation in estimated_orientations]
    roll_estimated = [orientation[0] for orientation in estimated_orientations]

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Orientation comparisons of Ground Truth and Estimated Positions")

    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Yaw")
    axs[0, 0].set_title("Yaw Ground Truth")
    axs[0, 0].plot(yaw_gt)

    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Yaw")
    axs[1, 0].set_title("Yaw Estimated")
    axs[1, 0].plot(yaw_estimated)

    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Pitch")
    axs[0, 1].set_title("Pitch Ground Truth")
    axs[0, 1].plot(pitch_gt)

    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Pitch")
    axs[1, 1].set_title("Pitch Estimated")
    axs[1, 1].plot(pitch_estimated)

    axs[0, 2].set_xlabel("Time")
    axs[0, 2].set_ylabel("Roll")
    axs[0, 2].set_title("Roll Ground Truth")
    axs[0, 2].plot(roll_gt)

    axs[1, 2].set_xlabel("Time")
    axs[1, 2].set_ylabel("Roll")
    axs[1, 2].set_title("Roll Estimated")
    axs[1, 2].plot(roll_estimated)

    fig.savefig("./hw3/imgs/orientation.png")


data, gt = read_mat("./hw3/data/studentdata0.mat")

map = Map()
# for tag in map.tags:
#     print(
#         f"{tag}- p1: {tuple(map.tags[tag].bottom_left)}, p2: {tuple(map.tags[tag].bottom_right)}, p3: {tuple(map.tags[tag].top_right)}, p4: {tuple(map.tags[tag].top_left)}"
#     )
positions: List[np.ndarray] = []
orientations: List[np.ndarray] = []
for datum in data:
    # Estimate the pose of the camera
    if len(datum.tags) == 0:
        continue
    orientation, position = map.estimate_pose(datum.tags)
    positions.append(position)
    orientations.append(orientation)

# Create multiplot and isometric plot
create_trajectory_plots(gt, positions)
create_orientation_plots(gt, orientations)
figure = plot_trajectory(
    [Coordinate(x=gti.x, y=gti.y, z=gti.z) for gti in gt], "Ground Truth"
)
figure.savefig("./hw3/imgs/ground_truth.png")
figure = plot_trajectory(
    [Coordinate(x=position[0], y=position[1], z=position[2]) for position in positions],
    "Estimated Trajectory",
)
figure.savefig("./hw3/imgs/estimated.png")
