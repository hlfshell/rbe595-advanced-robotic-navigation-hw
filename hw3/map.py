from typing import Dict, List, Optional

import numpy as np
from cv2 import solvePnP
from world import AprilTag, Coordinate


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
        for tag in tags:
            world_points.append(
                [
                    tuple(self.tags[tag.id].bottom_left),
                    tuple(self.tags[tag.id].bottom_right),
                    tuple(self.tags[tag.id].top_right),
                    tuple(self.tags[tag.id].top_left),
                ]
            )
        world_points = np.array(world_points)

        image_points = np.array(
            [
                [
                    tuple(tag.bottom_left),
                    tuple(tag.bottom_right),
                    tuple(tag.top_right),
                    tuple(tag.top_left),
                ]
                for tag in tags
            ]
        )

        _, orientation, position = solvePnP(
            world_points,
            image_points,
            self.camera_matrix,
            self.distortion_coefficients,
            flags=0,
        )

        return orientation, position
