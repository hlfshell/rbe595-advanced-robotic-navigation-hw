from __future__ import annotations
from cv2 import solvePnP
import numpy as np

import matplotlib.pyplot as plt

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from scipy.io import loadmat


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
            y_offset = 0.152 * row_index * 2
            if row_index + 1 >= 3:
                y_offset += extra_offset
            if row_index + 1 >= 6:
                y_offset += extra_offset

            for tag_index, tag in enumerate(row):
                x_offset = 0.152 * tag_index * 2

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

    def estimate_pose(self, tags: List[AprilTag]):
        """
        estimate_pose will, given a list of observed AprilTags,
        pair them with their real world coordinates in order to
        estimate the orientation and position of the camera at
        that moment in time.
        """
        world_points = []
        # for tag in tags:
        #     world_points.append(
        #         [
        #             tuple(self.tags[tag.id].bottom_left),
        #             tuple(self.tags[tag.id].bottom_right),
        #             tuple(self.tags[tag.id].top_right),
        #             tuple(self.tags[tag.id].top_left),
        #         ]
        #     )
        # world_points = np.array(world_points)
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

        # image_points = np.array(
        #     [
        #         [
        #             tuple(tag.bottom_left),
        #             tuple(tag.bottom_right),
        #             tuple(tag.top_right),
        #             tuple(tag.top_left),
        #         ]
        #         for tag in tags
        #     ]
        # )

        _, orientation, position = solvePnP(
            world_points,
            image_points,
            self.camera_matrix,
            self.distortion_coefficients,
            flags=0,
        )

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


def plot_trajectory(trajectory: List[Coordinate], title: str = "") -> None:
    """
    Given a list of coordinates as a trajectory plot the
    trajectory in 3D space.
    """

    x = [coord.x for coord in trajectory]
    y = [coord.y for coord in trajectory]
    z = [coord.z for coord in trajectory]

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes(projection="3d")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.dist = 11
    axes.set_title(title)

    axes.scatter3D(x, y, z, c=z, linewidths=0.5)

    return figure


data, gt = read_mat("./hw3/data/studentdata0.mat")

# Plot the trajectory of the ground truth
figure = plot_trajectory(
    [Coordinate(x=gti.x, y=gti.y, z=gti.z) for gti in gt],
    title="Ground Truth Trajectory",
)
figure.savefig("./test.png")

map = Map()
positions = []
for datum in data:
    # Estimate the pose of the camera
    if len(datum.tags) == 0:
        continue
    orientation, position = map.estimate_pose(datum.tags)
    print(position)
    positions.append(position)

figure = plot_trajectory(
    [Coordinate(x=position[0], y=position[1], z=position[2]) for position in positions],
    title="Estimated Trajectory via Camera",
)
figure.savefig("./test2.png")
