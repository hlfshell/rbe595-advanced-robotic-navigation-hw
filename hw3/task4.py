from typing import List, Optional, Tuple

import numpy as np
from world import Data, GroundTruth, Map, orientation_to_yaw_pitch_roll, read_mat
from task3 import interpolate_ground_truth

from cv2 import Rodrigues


# Covariance matrix below is calculated in task3.py through
# the analyzing the data of the drone's experiments
calculated_covariance_matrix = np.array(
    [
        [
            7.09701409e-03,
            2.66809900e-05,
            1.73906943e-03,
            4.49014777e-04,
            3.66195490e-03,
            8.76154421e-04,
        ],
        [
            2.66809900e-05,
            4.70388499e-03,
            -1.33432420e-03,
            -3.46505064e-03,
            1.07454548e-03,
            -1.69184839e-04,
        ],
        [
            1.73906943e-03,
            -1.33432420e-03,
            9.00885499e-03,
            1.80220246e-03,
            3.27846190e-03,
            -1.11786368e-03,
        ],
        [
            4.49014777e-04,
            -3.46505064e-03,
            1.80220246e-03,
            5.27060654e-03,
            1.01361187e-03,
            -5.86487142e-04,
        ],
        [
            3.66195490e-03,
            1.07454548e-03,
            3.27846190e-03,
            1.01361187e-03,
            7.24994152e-03,
            -1.36454993e-03,
        ],
        [
            8.76154421e-04,
            -1.69184839e-04,
            -1.11786368e-03,
            -5.86487142e-04,
            -1.36454993e-03,
            1.21162646e-03,
        ],
    ]
)


class UKF:

    def __init__(
        self,
        covariance_matrix: Optional[np.ndarray] = None,
    ):

        if covariance_matrix is None:
            self.covariance_matrix = calculated_covariance_matrix
        else:
            self.covariance_matrix = covariance_matrix

        self.last_state: np.ndarray = np.zeros((15, 1))
        self.previous_state_timestamp: float = 0.0

        # n is the number of dimensions for our given state
        self.n = 15
        # The number of sigma points we do are typically 2*n + 1,
        # so therefore...
        self.number_of_sigma_points = 2 * self.n + 1  # 31
        # kappa is a tuning value for the filter
        self.kappa = 0.1

        self.measurement_noise = np.identity(self.n) * 1e-3

        # ng and na are the biases from the IMU
        self.ng = np.zeros((3, 1))
        self.na = np.zeros((3, 1))

        # Create our state vector and set it to 0 - for the very first
        # step we need to initialize it to a non zero value
        self.mu = np.zeros((self.n, 1))

        # Sigma is our uncertainty matrix
        self.sigma = np.zeros((self.n, self.n))

        self.map = Map()

    def find_sigma_points(self, mu: np.ndarray) -> np.ndarray:
        number_of_points = mu.shape[0]

        # Based on our system, we expect this to be a 15x31 matrix
        sigma_points = np.zeros((number_of_points, self.number_of_sigma_points))

        # Set the first column of sigma points to be mu, since that is the mean
        # of our distribution (our center point)
        sigma_points[:, 0] = mu.reshape((15,))

        # Square root via Cholesky
        # sigma = np.linalg.cholesky(
        #     (number_of_points + self.kappa) * self.sigma
        # )  # <<<< is this right?
        sigma = np.sqrt((number_of_points + self.kappa) * self.sigma)
        # Should it be the covariance matrix instead??

        # Now for each point that we wish to go through for our sigma points we
        # move back and forth around the central point; thus we add, then
        # subtract the delta to find symmetrical points. We skip the first point
        # since we already set it to mu
        for i in range(0, number_of_points):
            sigma_points[:, i + 1] = mu.reshape((15,)) + sigma[:, i - 1]
            sigma_points[:, i + number_of_points + 1] = (
                mu.reshape((15,)) - sigma[:, i - 1]
            )

        return sigma_points

    def process_model(
        self, state: np.ndarray, delta_t: float, uw: np.ndarray, ua=np.ndarray
    ) -> np.ndarray:
        """
        Given a new state, and the amount of time that has passed between this one
        and the prior one, the gyro and accelerometer readings, then return the
        resulting state for the time delta.

        Our goal is to form xdot, which is
                | pdot          |
                | G(q)^-1 u_w   |
        xdot =  | g + R(q) u_a  |
                | n_g           |
                | n_a           |

        where pdot is the derivative of the position
        G(q) is the rotation matrix from the world frame to the drone frame, u_w is
            the force vector
        R(q) is the rotation matrix from the drone frame to the world frame, u_a is
            the acceleration vector (g is gravity, negative due to frame orientation)
        n_g is the bias from the gyroscope
        n_a is the bias from the accelerometer
        """
        xdot = np.zeros((15, 1))
        g = -9.81  # good ol' gravity

        # Extract the position, orientation, and velocities from the state
        # position = state[0:3]
        orientation = state[3:6]
        velocities = state[6:9]

        # Create our rotation matrix from the drone frame to the world frame
        # via our orientation roll pitch yaw
        G_q, _ = Rodrigues(orientation)
        R_q = np.linalg.inv(G_q)

        # u_w is the gyroscope measurement, and u_a is the accelerometer
        # measurement from our data point
        xdot[0:3] = velocities
        xdot[3:6] = np.dot(G_q.T, uw)
        xdot[6:9] = np.array([0, 0, g]).reshape((3, 1)) + np.dot(
            R_q, ua
        )  # np.array([0, 0, g]) + np.dot(R_q, ua)
        xdot[9:12] = self.ng
        xdot[12:15] = self.na

        return xdot

    def update(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        pass

    def predict(self, mu: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
        pass

    def rmse(
        self, gts: List[GroundTruth], estimated_positions: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a set of ground truth points, return the total RMSE between the groundtruth
        and estimated positions. The return is the error in the position, and then the
        orientation.
        """
        for estimate in estimated_positions:
            gt = interpolate_ground_truth(gts, estimate)
        # TODO
        pass

    def __data_to_vector(self, data: Data) -> np.ndarray:
        """
        Convert our helpful data object to a numpy array after estimating
        pose, orientation, and velocities
        """
        vector = np.zeros((15, 1))

        # Estimate our pose
        position, orientation = self.map.estimate_pose(data.tags)
        vector[0:3] = position
        vector[3:6] = orientation_to_yaw_pitch_roll(orientation)

        # Calculate the velocities/deltas since the last position
        delta_position = position - self.last_state[0:3]
        delta_time = data.timestamp - self.previous_state_timestamp
        vector[6:9] = delta_position / delta_time

        vector[9:12] = self.ng
        vector[12:15] = self.na

    def run(self, estimated_positions: List[Data]) -> List[np.ndarray]:
        """
        Given a set of estimated positions, return the estimated positions after running
        the Unscented Kalman Filter over the data
        """
        self.filtered_positions: List[np.ndarray] = []

        # First we need to initialize our initial position to the 0th estimated
        # position
        self.mu = self.__data_to_vector(estimated_positions[0])
        # Note that any velocities should be 0 for the first step
        self.mu[6:9] = np.zeros((3, 1))

        for index, data in enumerate(estimated_positions):
            # Skip the 0th estimated position
            if index == 0:
                continue

            # Grab the current state vector for our given estimated position
            estimated_state = self.__data_to_vector(data)

            # What is the current accelerometer and gyroscope readings?
            uw = data.rpy
            ua = data.acc

            # Delta t since our lsat prediction
            delta_t = data.timestamp - self.previous_state_timestamp

            # Get our sigma points
            sigma_points = self.find_sigma_points(self.mu)

            # Use the process model to perform the state transition off
            # of our current estimate for each sigma point. Since we have
            # (self.n * 2) + 1 = 31 sigma points, we expect this to be a
            # (15,31) matrix
            transitioned = np.zeros((15, 31))
            for sigma_point in sigma_points:
                # Predict the new state for each sigma point
                transitioned[:, sigma_point] = self.process_model(
                    sigma_point, delta_t, uw, ua
                )

            # Run the prediction step based off of our state transition
            pass

            # Run the update step to filter our estimated position and resulting
            # sigma (mu and sigma)
            pass

            # Save the estimated position and timestamp
            pass

        return filtered_positions


x = UKF()
# print(x.find_sigma_points(np.ones((15, 1))))

print(x.process_model(np.ones((15, 1)), 0.1, np.ones((3, 1)) * 2, np.ones((3, 1)) * 3))
