from typing import List, Optional, Tuple

import numpy as np
from world import (
    Coordinate,
    Data,
    GroundTruth,
    Map,
    orientation_to_yaw_pitch_roll,
    read_mat,
    plot_trajectory,
)

# from task3 import interpolate_ground_truth

from cv2 import Rodrigues

from time import time

from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
from math import pi

from utils import interpolate_ground_truth

from plot import plot_rmse_loss


# Covariance matrix below is calculated in task3.py through
# the analyzing the data of the drone' experiments
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
        measurement_covariance_matrix: Optional[np.ndarray] = None,
        # kappa: float = 0.1,
        # alpha: float = 1e-3,
        # beta: float = 2.0,
        kappa: float = 1,
        alpha: float = 1,
        beta: float = 2.0,
    ):

        if measurement_covariance_matrix is None:
            self.measurement_covariance_matrix = calculated_covariance_matrix
        else:
            self.measurement_covariance_matrix = measurement_covariance_matrix

        # n is the number of dimensions for our given state
        self.n = 15

        self.last_state: np.ndarray = np.zeros((self.n, 1))

        # The number of sigma points we do are typically 2*n + 1,
        # so therefore...
        self.number_of_sigma_points = 2 * self.n + 1  # 31
        # kappa is a tuning value for the filter
        self.kappa = kappa
        # alpha is used to calculate initial covariance weights
        self.alpha = alpha
        self.beta = beta

        self.measurement_noise = np.identity(self.n) * 1e-3

        # ng and na are the biases from the IMU
        self.ng = np.zeros((3, 1))
        self.na = np.zeros((3, 1))

        # Create our state vector and set it to 0 - for the very first
        # step we need to initialize it to a non zero value
        self.mu = np.zeros((self.n, 1))

        # Sigma is our uncertainty matrix
        # self.sigma = np.zeros((self.n, self.n))

        self.map = Map()

        # Calculate our lambda and weights for use throughout the filter
        self.λ = self.alpha**2 * (self.n + self.kappa) - self.n
        print("λ", self.λ)

        # We have three weights - the 0th mean weight, the 0th covariance
        # weight, and weight_i, which is the ith weight for all other
        # mean and covariance calculations (equivalent)

        # The 0th weight for the mean is
        #    λ
        # ---------
        # n + λ
        weights_mean_0 = self.λ / (self.n + self.λ)

        # The 0th weight for the covariance is
        #     λ
        # --------- + 1 - alpha^2 + beta
        # n + λ
        weights_covariance_0 = (
            (self.λ / (self.n + self.λ)) + (1 - (self.alpha**2)) + self.beta
        )

        # The remaining weights for mean and covariance are equivalent, at:
        #     1
        # -----------
        # 2(n + λ)
        weight_i = 1 / (2 * (self.n + self.λ))

        self.weights_mean = np.zeros((self.number_of_sigma_points))
        self.weights_covariance = np.zeros((self.number_of_sigma_points))
        self.weights_mean[0] = weights_mean_0
        self.weights_mean[1:] = weight_i
        self.weights_covariance[0] = weights_covariance_0
        self.weights_covariance[1:] = weight_i

    def find_sigma_points(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        # Based on our system, we expect this to be a 15x31 matrix
        # sigma_points = np.zeros((number_of_points, self.number_of_sigma_points))
        sigma_points = np.zeros((self.number_of_sigma_points, self.n, 1))

        # Set the first column of sigma points to be mu, since that is the mean
        # of our distribution (our center point)
        sigma_points[0] = mu

        # Square root via Cholesky
        # sigma = np.linalg.cholesky((self.n + self.λ) * sigma)
        # sigma_sqrt = np.sqrt((self.number_of_sigma_points + self.λ) * sigma)
        # sigma_sqrt = sqrtm((self.number_of_sigma_points + self.λ) * sigma)

        # # We need S to be a 15x15, and it is currently a 6x6, so we will
        # # place it in the upper lefthand corner of a 15x15
        # S = np.zeros((self.n, self.n))
        # # S[0:6, 0:6] = sigma_sqrt
        # S = sigma_sqrt
        S = sqrtm((self.n + self.kappa) * sigma)

        # Now for each point that we wish to go through for our sigma points we
        # move back and forth around the central point; thus we add, then
        # subtract the delta to find symmetrical points. We skip the first point
        # since we already set it to mu
        # Julier Sigma Point Method
        for i in range(self.n):
            sigma_points[i + 1] = mu + S[i].reshape((15, 1))
            sigma_points[self.n + i + 1] = mu - S[i].reshape((15, 1))

        # scaling_params = np.zeros((self.n, 1))
        # for i in range(0, self.n):
        # for i in range(1, self.n + 1):
        #     # REDO THIS TO BE COLUMN centric
        #     # sigma_points[:, i + 1] = mu.reshape((15,)) + S[:, i - 1]
        #     # sigma_points[:, i + self.number_of_sigma_points + 1] = (
        #     #     mu.reshape((15,)) - S[:, i - 1]
        #     # )
        #     # sigma_points[i + 1, :] = mu + S[i, :].reshape((15, 1))
        #     # sigma_points[i + self.n + 1, :] = mu - S[i, :].reshape((15, 1))
        #     scaling_params = S[0 : self.n, i - 1].reshape((15, 1))
        #     # sigma_points[0 : self.n, i] = mu[0 : self.n] + scaling_params
        #     # sigma_points[0 : self.n, self.n + i] = mu[0 : self.n] - scaling_params
        #     sigma_points[i] = mu[0 : self.n] + scaling_params
        #     sigma_points[self.n + i] = mu[0 : self.n] - scaling_params

        # Plot the sigma points
        # figure = plot_trajectory(
        #     [
        #         Coordinate(x=position[0], y=position[1], z=position[2])
        #         for position in sigma_points
        #     ],
        #     "Sigma Points",
        # )
        # figure.savefig("./sigma_points.png")

        return sigma_points

    def process_model(
        self, state: np.ndarray, delta_t: float, ua: np.ndarray, uw=np.ndarray
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
        # Get roll pitch yaw from the state
        roll = orientation[0][0]
        pitch = orientation[1][0]
        yaw = orientation[2][0]
        # G_q, _ = Rodrigues(orientation)
        # R_q = np.linalg.inv(G_q)

        G_q = np.array(
            [
                [np.cos(pitch), 0.0, -np.cos(roll) * np.sin(pitch)],
                [0.0, 1.0, np.sin(roll)],
                [np.sin(pitch), 0.0, np.cos(roll) * np.cos(pitch)],
            ]
        )
        R_q = np.array(
            [
                [
                    np.cos(yaw) * np.cos(pitch)
                    - np.sin(roll) * np.sin(yaw) * np.sin(pitch),
                    -np.cos(roll) * np.sin(yaw),
                    np.cos(yaw) * np.sin(pitch)
                    + np.cos(pitch) * np.cos(roll) * np.sin(yaw),
                ],
                [
                    np.cos(pitch) * np.sin(yaw)
                    + np.cos(yaw) * np.sin(roll) * np.sin(pitch),
                    np.cos(roll) * np.cos(yaw),
                    np.sin(yaw) * np.sin(pitch)
                    - np.cos(yaw) * np.cos(pitch) * np.sin(roll),
                ],
                [
                    -np.cos(roll) * np.sin(pitch),
                    np.sin(roll),
                    np.cos(roll) * np.cos(pitch),
                ],
            ]
        )

        # u_w is the gyroscope measurement, and u_a is the accelerometer
        # measurement from our data point
        xdot[0:3] = velocities
        # xdot[3:6] = np.dot(G_q.T, uw.reshape((3, 1)))
        xdot[3:6] = np.dot(np.linalg.inv(G_q), uw.reshape((3, 1)))
        xdot[6:9] = np.array([0, 0, g]).reshape((3, 1)) + np.dot(
            R_q, ua.reshape((3, 1))
        )
        # xdot[9:12] = self.ng
        # xdot[12:15] = self.na
        xdot[9:12] = state[9:12]
        xdot[12:15] = state[12:15]

        # Calculate the new state with our xdot
        return state + (xdot * delta_t)

    def process_model2(
        self, state: np.ndarray, delta_t: float, ua: np.ndarray, uw=np.ndarray
    ) -> np.ndarray:
        """
        Given a set of particles, perform the state transition across
        all of them.

        Given the original set of particles, a delta t, and experienced
        accelerations from the IMU, accelerations from the gyroscope,
        figure out the resulting change of each particle (velocity, etc)
        and calculate:

                | pdot          |
                | G(q)^-1 u_w   |
        xdot =  | g + R(q) u_a  |
                | n_g           |
                | n_a           |

        ...per particle, resulting in a self.particles x 15 matrix.

        ...where:
        pdot is the derivative of the position
        G(q) is the rotation matrix from the world frame to the drone frame, u_w is
            the force vector
        R(q) is the rotation matrix from the drone frame to the world frame, u_a is
            the acceleration vector (g is gravity, negative due to frame orientation)
        n_g is the bias from the gyroscope
        n_a is the bias from the accelerometer
        """

        xdot = np.zeros((15, 1))
        # ua and uw are the bias from the accelerometer and gyroscope
        # respectively.
        ua = ua.reshape((3, 1))
        uw = uw.reshape(3, 1)
        g = -9.81

        # Extract the orientation, and velocities from the state
        orientations = state[3:6]
        velocities = state[6:9]

        # Create our rotation matrix from the drone frame to the world frame
        # via our orientation roll pitch yaw. We do this by finding G_q and
        # then solving G_q's inverse. The inverse is a when doing it
        # vectorized w/ all particles at once; so we quickly just manually
        # use the analytical result off the bat.
        theta = orientations[0]
        phi = orientations[1]
        psi = orientations[2]

        G_q_inv = np.zeros((3, 3, 1))
        G_q_inv[0, 0] = np.cos(theta)
        # G_q_inv[0, 1] = 0.0
        G_q_inv[0, 2] = np.sin(theta)
        G_q_inv[1, 0] = np.sin(phi) * np.sin(theta) / np.cos(phi)
        G_q_inv[1, 1] = 1.0
        G_q_inv[1, 2] = -np.cos(theta) * np.sin(phi) / np.cos(phi)
        G_q_inv[2, 0] = -np.sin(theta) / np.cos(phi)
        # G_q_inv[2, 1] = 0.0
        G_q_inv[2, 2] = np.cos(theta) / np.cos(phi)

        # We create R_q, again manually, to keep things a bit easier
        R_q = np.zeros((3, 3, 1))
        R_q[0, 0] = np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(phi) * np.sin(
            theta
        )
        R_q[0, 1] = -np.cos(phi) * np.sin(psi)
        R_q[0, 2] = np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(
            psi
        )
        R_q[1, 0] = np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(
            theta
        )
        R_q[1, 1] = np.cos(phi) * np.cos(psi)
        R_q[1, 2] = np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(
            phi
        )
        R_q[2, 0] = -np.cos(phi) * np.sin(theta)
        R_q[2, 1] = np.sin(phi)
        R_q[2, 2] = np.cos(phi) * np.cos(theta)

        xdot[0:3] = velocities

        xdot[3:6] = np.dot(G_q_inv.reshape((3, 3)), uw.reshape((3, 1)))
        xdot[6:9] = np.array([0, 0, g]).reshape((3, 1)) + np.dot(
            R_q.reshape((3, 3)), ua.reshape((3, 1))
        )

        # We are ignoring this for now; it's defaulted to zero so no need
        # to do the zeroing again.
        # xdot[9:12] = np.zeros((3, 1))
        # xdot[12:15] = np.zeros((3, 1))

        # Add our xdot delta to our particles
        state = state + (xdot * delta_t)

        return state

    def update(
        self, state: np.ndarray, mu: np.ndarray, sigma: np.ndarray, sigma_points
    ) -> np.ndarray:
        """
        update takes the state, mubar, and sigmabar and performs the
        update step
        """
        # Use the generated mu and sigma from the predict function (mubar and
        # sigmabar) to find new sigma points
        # sigma_points = self.find_sigma_points(mu, sigma)

        # Apply the measurement function across each new sigma point
        measurement_points = np.zeros_like(sigma_points)
        for i in range(self.number_of_sigma_points):
            measurement_points[i] = self.measurement_function(sigma_points[i])

        # Calculate the mean of the measurement points by their respective
        # weights. The weights have a 1/N term so the mean is calculated
        # through their addition
        zhat = np.zeros((self.n, 1))
        for i in range(0, self.number_of_sigma_points):
            zhat += self.weights_mean[i] * measurement_points[i]

        R = np.zeros((self.n, self.n))
        R[0:6, 0:6] = np.diag(self.measurement_covariance_matrix)
        St = np.zeros((self.n, self.n))
        differences_z = measurement_points - zhat
        for i in range(0, self.number_of_sigma_points):
            St += self.weights_covariance[i] * np.dot(
                differences_z[i], differences_z[i].T
            )
        St += R
        # St = self.fix_covariance(St)

        # Find the cross-covariance
        # Find the differences between the generated sigma points from
        # earlier and mu
        sigmahat_t = np.zeros((self.n, self.n))
        differences_x = sigma_points - mu
        for i in range(0, self.number_of_sigma_points):
            sigmahat_t += self.weights_covariance[i] * np.dot(
                differences_x[i], differences_z[i].T
            )
        # sigmahat_t = self.fix_covariance(sigmahat_t)

        # kalman_gain = np.dot(sigmahat_t, np.linalg.inv(St))
        kalman_gain = np.dot(sigmahat_t, np.linalg.pinv(St))

        # Update the mean and covariance
        current_position = mu + np.dot(kalman_gain, state - zhat)
        covariance = sigma - np.dot(kalman_gain, St).dot(kalman_gain.T)
        covariance = self.fix_covariance(covariance)

        return current_position, covariance

    def fix_covariance(self, covariance: np.ndarray, jitter: float = 1e-3):
        """
        Fix the covariance matrix to be positive definite with the
        jitter method on its eigen values. Will continually add more
        jitter until the matrix is symmetric positive definite.
        """
        # Is it symmetric?
        symmetric = np.allclose(covariance, covariance.T)
        # Is it positive definite?
        try:
            np.linalg.cholesky(covariance)
            positive_definite = True
        except np.linalg.LinAlgError:
            positive_definite = False

        # If the result is symmetric and positive definite, return it
        if symmetric and positive_definite:
            return covariance

        # Make covariance matrix symmetric
        covariance = (covariance + covariance.T) / 2

        # Set the eigen values to zero
        eig_values, eig_vectors = np.linalg.eig(covariance)
        eig_values[eig_values < 0] = 0
        eig_values += jitter

        # Reconstruct the matrix
        covariance = eig_vectors.dot(np.diag(eig_values)).dot(eig_vectors.T)

        return self.fix_covariance(covariance, jitter=10 * jitter)

    def measurement_function(self, state: np.ndarray) -> np.ndarray:
        """
        measurement_function takes the current state and returns the measurement
        of the state adjusted by our measurement covariance matrix noise

        Note that here state is (6,1), not (15,1), as we are essentially measuring
        6 measurements for the state - the solvePnP positions and orientations.
        """
        # noise_adjustment = np.diag(self.measurement_covariance_matrix).reshape(6, 1)
        # # Extend it to be a 15,1 to match the state
        # noise_adjustment = np.vstack((noise_adjustment, np.zeros((9, 1))))

        # return state + noise_adjustment
        noise_adjustment = np.zeros((self.n, 1))
        c = np.zeros((6, self.n))
        c[0:6, 0:6] = np.eye(6)

        R = np.diag(self.measurement_covariance_matrix).reshape(6, 1)
        noise_adjustment[0:6] = np.dot(c, state) + R

        return noise_adjustment

    def predict(
        self, sigma_points: np.ndarray, ua: np.ndarray, uw: np.ndarray, delta_t: float
    ) -> Tuple[np.ndarray]:
        """
        predict takes the current sigma points for the estimate and performs
        the state transition across them. We then compute the mean and the
        covariance of the resulting transformed sigma points.
        """
        # For each sigma point, run them through our state transition function
        transitioned_points = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            transitioned_points[i, :] = self.process_model2(
                sigma_points[i], delta_t, uw, ua
            )

        # Calculate the mean of the transitioned points by their respective
        # weights. Since we included a 1/N term in the weights, adding them
        # together effectively moves us towards a weighted mean
        mu = np.zeros((self.n, 1))
        for i in range(0, self.number_of_sigma_points):
            mu += self.weights_mean[i] * transitioned_points[i]

        # Calculate the covariance of the transitioned points by their
        # respective weights. As before, the weights contain a 1/N term so
        # we are effectively finding the average. We expect a NxN output
        # for our sigma matrix
        Q = np.random.normal(scale=5e-1, size=(15, 15))
        differences = transitioned_points - mu
        sigma = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            sigma += self.weights_covariance[i] * np.dot(
                differences[i], differences[i].T
            )
        sigma += Q
        # Return sigma to 6x6 by just taking the upper left corner
        # sigma = sigma[0:6, 0:6]
        # print(mu.shape)
        # print(sigma.shape)
        # mu = np.mean(mu, axis=0)
        # sigma = np.mean(sigma, axis=0)
        # print(mu.shape)
        # print(sigma.shape)
        # raise "dead"

        return mu, sigma, transitioned_points

    def rmse(
        self, gts: List[GroundTruth], estimated_positions: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a set of ground truth points, return the total RMSE between the groundtruth
        and estimated positions. The return is the error in the position, and then the
        orientation.
        """
        position_errors = np.zeros((len(estimated_positions), 1))
        orientation_errors = np.zeros((len(estimated_positions), 1))

        for index, estimate in enumerate(estimated_positions):
            gt = interpolate_ground_truth(gts, estimate)
            gt_position = gt[0:3]
            gt_orientation = gt[3:6]

            estimate_position = estimate[0:3]
            estimate_orientation = estimate[3:6]

            estimate_error = np.sqrt(np.mean((gt_position - estimate_position) ** 2))
            orientation_error = np.sqrt(
                np.mean((gt_orientation - estimate_orientation) ** 2)
            )

            position_errors[index] = estimate_error
            orientation_errors[index] = orientation_error

        return position_errors, orientation_errors

    def __data_to_vector(
        self, data: Data, prior_state: np.ndarray, last_timestamp: float
    ) -> np.ndarray:
        """
        Convert our helpful data object to a numpy array after estimating
        pose, orientation, and velocities
        """
        vector = np.zeros((15, 1))

        # Estimate our pose
        orientation, position = self.map.estimate_pose(data.tags)

        vector[0:3] = np.array(position).reshape((3, 1))
        vector[3:6] = np.array(orientation_to_yaw_pitch_roll(orientation)).reshape(
            (3, 1)
        )

        # Calculate the velocities/deltas since the last position
        # delta_position = vector[0:3] - prior_state[0:3]
        # delta_time = data.timestamp - last_timestamp
        # vector[6:9] = delta_position / delta_time

        # vector[9:12] = data.rpy.reshape((3, 1))
        # vector[12:15] = data.acc.reshape((3, 1))

        return vector

    def run(self, estimated_positions: List[Data]) -> List[np.ndarray]:
        """
        Given a set of estimated positions, return the estimated positions after running
        the Unscented Kalman Filter over the data
        """
        filtered_positions: List[np.ndarray] = []

        # First we need to initialize our initial position to the 0th estimated
        # position
        state = self.__data_to_vector(
            estimated_positions[0], np.zeros((self.n, 1)), 0.0
        )
        # Note that any velocities should be 0 for the first step
        state[6:9] = np.zeros((3, 1))
        previous_state_timestamp = estimated_positions[0].timestamp

        process_covariance_matrix = np.eye(15) * 1e-3

        def coords(x):
            return (x[0][0], x[1][0], x[2][0])

        print(0, coords(state))

        for index, data in enumerate(estimated_positions):
            # Skip the 0th estimated position
            if index == 0:
                continue
            # if index == 8:
            #     raise "deaded"
            # print(index)

            # Grab the current state vector for our given estimated position
            state = self.__data_to_vector(data, state, previous_state_timestamp)
            # print(index, coords(state))

            # What is the current accelerometer and gyroscope readings?
            ua = data.acc
            uw = data.omg

            # Delta t since our last prediction
            delta_t = data.timestamp - previous_state_timestamp
            previous_state_timestamp = data.timestamp

            # Get our sigma points. We expect (self.n * 2) + 1 = 31 sigma points
            # for a (15,31) matrix
            sigma_points = self.find_sigma_points(state, process_covariance_matrix)

            # print("mu/state", state)
            # print("sigma/ process covariance", process_covariance_matrix)
            # Run the prediction step based off of our state transition
            mubar, sigmabar, transitioned_points = self.predict(
                sigma_points, ua, uw, delta_t
            )

            # print("mubar", mubar)
            # print("sigmabar", sigmabar)
            # Run the update step to filter our estimated position and resulting
            # sigma (mu and sigma)
            mu, sigma = self.update(state, mubar, sigmabar, transitioned_points)

            # print("post update mu", mu)
            # print("post update sigma", sigma)
            # print("===")
            # Our current position is mu, and our new process covariance
            # matrix is sigma
            process_covariance_matrix = sigma
            state = mu

            filtered_positions.append(state)

        return filtered_positions


def create_overlay_plots(
    ground_truth: List[GroundTruth],
    estimated_positions: List[np.ndarray],
    estimated_orientations: List[np.ndarray],
    estimated_times: List[float],
    dataset_name: str,
):
    gt_coordinates = [Coordinate(x=gti.x, y=gti.y, z=gti.z) for gti in ground_truth]
    estimated_coordinates = [
        Coordinate(x=position[0], y=position[1], z=position[2])
        for position in estimated_positions
    ]

    x_gt = [coord.x for coord in gt_coordinates]
    y_gt = [coord.y for coord in gt_coordinates]
    z_gt = [coord.z for coord in gt_coordinates]
    gt_times = [gti.timestamp for gti in ground_truth]

    x_estimated = [coord.x for coord in estimated_coordinates]
    y_estimated = [coord.y for coord in estimated_coordinates]
    z_estimated = [coord.z for coord in estimated_coordinates]

    yaw_gt = [gti.yaw for gti in ground_truth]
    pitch_gt = [gti.pitch for gti in ground_truth]
    roll_gt = [gti.roll for gti in ground_truth]

    yaw_estimated = [orientation[2] for orientation in estimated_orientations]
    pitch_estimated = [orientation[1] for orientation in estimated_orientations]
    roll_estimated = [orientation[0] for orientation in estimated_orientations]

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle("Orientation Comparisons of Ground Truth and Estimated Positions")

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Yaw")
    axs[0].set_title("Yaw")
    axs[0].set_ylim(-pi / 2, pi / 2)
    axs[0].plot(gt_times, yaw_gt, label="Ground Truth")
    axs[0].plot(estimated_times, yaw_estimated, label="Estimated")

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Pitch")
    axs[1].set_title("Pitch")
    axs[1].set_ylim(-pi / 2, pi / 2)
    axs[1].plot(gt_times, pitch_gt, label="Ground Truth")
    axs[1].plot(estimated_times, pitch_estimated, label="Estimated")

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Roll")
    axs[2].set_title("Roll")
    axs[2].set_ylim(-pi / 2, pi / 2)
    axs[2].plot(gt_times, roll_gt, label="Ground Truth")
    axs[2].plot(estimated_times, roll_estimated, label="Estimated")

    fig.savefig(f"./hw3/imgs/task1_2/{dataset_name}_orientation_merged.png")

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle("Trajectory Comparisons of Ground Truth and Estimated Positions")

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Top-Down")
    axs[0].scatter(x_gt, y_gt, c=z_gt, label="Ground Truth")
    axs[0].scatter(x_estimated, y_estimated, c=z_estimated, label="Estimated")

    axs[1].set_xlabel("Y")
    axs[1].set_ylabel("Z")
    axs[1].set_title("Side X View")
    axs[1].scatter(y_gt, z_gt, c=z_gt, label="Ground Truth")
    axs[1].scatter(y_estimated, z_estimated, c=z_estimated, label="Estimated")

    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Z")
    axs[2].set_title("Side Y View")
    axs[2].scatter(x_gt, z_gt, c=z_gt, label="Ground Truth")
    axs[2].scatter(x_estimated, z_estimated, c=z_estimated, label="Estimated")

    fig.savefig(f"./hw3/imgs/task4/{dataset_name}_trajectory_merged.png")

    gt_coords = [Coordinate(x=gti.x, y=gti.y, z=gti.z) for gti in ground_truth]
    estimated_coords = [
        Coordinate(x=position[0], y=position[1], z=position[2])
        for position in estimated_positions
    ]

    fig = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes(projection="3d")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.dist = 11
    axes.set_title("Ground Truth and Estimated Positions Isometric View")

    axes.scatter3D(
        [coord.x for coord in gt_coords],
        [coord.y for coord in gt_coords],
        [coord.z for coord in gt_coords],
        c=[coord.z for coord in gt_coords],
        linewidths=0.5,
        label="Ground Truth",
    )

    axes.scatter3D(
        [coord.x for coord in estimated_coords],
        [coord.y for coord in estimated_coords],
        [coord.z for coord in estimated_coords],
        c=[coord.z for coord in estimated_coords],
        linewidths=0.5,
        label="Estimated",
    )

    fig.savefig(f"./hw3/imgs/task4/{dataset_name}_isometric.png")


if __name__ == "__main__":
    x = UKF()

    dataset = "./hw3/data/studentdata3.mat"

    base_data, gt = read_mat(dataset)

    map = Map()

    positions: List[np.ndarray] = []
    orientations: List[np.ndarray] = []
    times: List[float] = []
    data: List[Data] = []
    interpolated_gt: List[np.ndarray] = []
    camera_estimations: List[np.ndarray] = []
    for datum in base_data:
        # Estimate the pose of the camera
        if len(datum.tags) == 0:
            continue
        # We do the try here because some datasets have the data
        # recording *before* the ground truth vicon is recording
        # and we want to ignore those points as we can't compare
        # it to anything later
        try:
            interpolated_gt.append(interpolate_ground_truth(gt, datum))
        except:
            continue
        data.append(datum)
        orientation, position = map.estimate_pose(datum.tags)
        camera_estimations.append(np.concatenate([position, orientation]))
        positions.append(position)
        orientations.append(orientation_to_yaw_pitch_roll(orientation))
        times.append(datum.timestamp)

    start = time()
    results = x.run(data)
    print(f"Time taken for UKF: {time() - start:.2f} seconds")

    print("results are", len(results), results[-1])
    for i in range(len(results) - 6, len(results)):
        print(i, (results[i][0][0], results[i][1][0], results[i][2][0]))

    fig = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes(projection="3d")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.dist = 11
    axes.set_title("Ground Truth and Estimated Positions Isometric View")
    gt_coords = [Coordinate(x=gt[i].x, y=gt[i].y, z=gt[i].z) for i in range(len(gt))]
    axes.scatter3D(
        [coord.x for coord in gt_coords],
        [coord.y for coord in gt_coords],
        [coord.z for coord in gt_coords],
        c=[coord.z for coord in gt_coords],
        linewidths=0.5,
        label="Ground Truth",
    )
    axes.scatter3D(
        [coord[0] for coord in results],
        [coord[1] for coord in results],
        [coord[2] for coord in results],
        c=[coord[2] for coord in results],
        linewidths=0.5,
        label="Estimated",
    )

    fig.savefig("./test_ukf_isometric.png")

    position_rmse, orientation_rmse = plot_rmse_loss(
        interpolated_gt,
        camera_estimations,
        results,
        times,
    )
    position_rmse.savefig("./test_ukf_position_rmse.png")
    orientation_rmse.savefig("./test_ukf_orientation_rmse.png")

    # Create trajectory plots from output
    # figure = plot_trajectory(
    #     [
    #         Coordinate(x=position[0], y=position[1], z=position[2])
    #         for position in results
    #     ],
    #     [Coordinate(x=gt[i].x, y=gt[i].y, z=gt[i].z) for i in range(len(gt))],
    #     "UKF Positions",
    # )
    # figure.savefig("./test_ukf_trajectory.png")

    # figure = plot_trajectory(
    #     [Coordinate(x=gt[i].x, y=gt[i].y, z=gt[i].z) for i in range(len(gt))],
    #     "Ground Truth",
    # )
    # figure.savefig("./test_ukf_gt_trajectory.png")
