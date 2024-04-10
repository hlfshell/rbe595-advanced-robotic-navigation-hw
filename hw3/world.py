from __future__ import annotations

from math import pi
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cv2 import Rodrigues, solvePnP
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
    # drpy: np.ndarray
    acc: np.ndarray
    omg: np.ndarray


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
                if tag_index >= 3:
                    y_offset += extra_offset
                if tag_index >= 6:
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
        # The resulting matrix converts coordinates in the camera frame to the
        # drone frame.
        # XYZ = [-0.04, 0.0, -0.03];
        # Yaw = pi/4;
        rotation_z = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        # Because the camera is pointing down, we are effectively rotated
        # pi radians about the x-axis
        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        )
        rotation = np.dot(rotation_x, rotation_z)

        # Combine it all with the offset as specified.
        camera_to_drone_frame = np.array(
            [
                [rotation[0, 0], rotation[0, 1], rotation[0, 2], -0.04],
                [rotation[1, 0], rotation[1, 1], rotation[1, 2], 0],
                [rotation[2, 0], rotation[2, 1], rotation[2, 2], -0.03],
                [0, 0, 0, 1],
            ]
        )

        # Convert the orientation to rotation matrix via Rodrigues' formula.
        # This is rvec from solvePnP to rotation matrix, representing the
        # rotation from the camera frame to the world frame. We combine it
        # with the position from solvePnP to get the full translation frame
        # of camera to world coordinates.
        orientation = Rodrigues(orientation)[0]

        camera_to_world_frame = np.array(
            [
                np.concatenate((orientation[0], position[0])),
                np.concatenate((orientation[1], position[1])),
                np.concatenate((orientation[2], position[2])),
                [0, 0, 0, 1],
            ]
        )

        # We aim to convert from the calculated camera position to the drone position.
        # To do this we multiply by the inverse of our world_to_camera_frame by our
        # drone_to_camera_frame, giving us a world to drone frame transformation.
        drone_to_world_frame = np.dot(
            np.linalg.inv(camera_to_world_frame), camera_to_drone_frame
        )

        position = drone_to_world_frame[0:3, 3]
        # Convert the rotation matrix back to a vector
        orientation = rotation_matrix_to_euler_angles(drone_to_world_frame[0:3, 0:3])

        return orientation, position


def rotation_matrix_to_euler_angles(
    rotation_matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """
    rotation_matrix_to_euler_angles converts a 3x3 rotation matrix to
    a tuple of Euler angles in XZY rotation order.
    """
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]

    yaw = np.arctan(-r12 / r22)
    roll = np.arctan(r32 * np.cos(yaw) / r22)
    pitch = np.arctan(-r31 / r33)

    return yaw, pitch, roll


def orientation_to_yaw_pitch_roll(
    orientation: np.ndarray,
) -> Tuple[float, float, float]:
    """
    orientation_to_yaw_pitch_roll will take a rotation matrix and
    convert it to a tuple of yaw, pitch, and roll.
    """
    # Convert the 3x1 matrix to a rotation matrix
    rotation = Rodrigues(orientation)[0]

    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    pitch = np.arctan2(
        -rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)
    )
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])

    return yaw, pitch, roll


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

        if "drpy" in datum:
            omg = datum["drpy"]
        else:
            omg = datum["omg"]

        data.append(
            Data(
                img=datum["img"],
                tags=tags,
                timestamp=datum["t"],
                rpy=datum["rpy"],
                # drpy=datum["drpy"],
                # drpy=0.0,
                acc=datum["acc"],
                omg=omg,
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
