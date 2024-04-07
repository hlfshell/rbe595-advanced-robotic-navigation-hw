from typing import List, Optional, Tuple

import numpy as np
from world import Data, GroundTruth, Map, orientation_to_yaw_pitch_roll, read_mat
from task3 import interpolate_ground_truth

from cv2 import Rodrigues


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
        kappa: float = 0.1,
        alpha: float = 1e-3,
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

    def find_sigma_points(
        self, mu: np.ndarray, sigma: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if sigma is None:
            sigma = self.measurement_covariance_matrix

        # Based on our system, we expect this to be a 15x31 matrix
        # sigma_points = np.zeros((number_of_points, self.number_of_sigma_points))
        sigma_points = np.zeros((self.number_of_sigma_points, self.n, 1))

        # Set the first column of sigma points to be mu, since that is the mean
        # of our distribution (our center point)
        sigma_points[0] = mu

        # Square root via Cholesky
        # sigma = np.linalg.cholesky((self.n + self.λ) * sigma)
        sigma_sqrt = np.sqrt((self.number_of_sigma_points + self.λ) * sigma)
        # We need S to be a 15x15, and it is currently a 6x6, so we will
        # place it in the upper lefthand corner of a 15x15
        S = np.zeros((self.n, self.n))
        S[0:6, 0:6] = sigma_sqrt

        # Now for each point that we wish to go through for our sigma points we
        # move back and forth around the central point; thus we add, then
        # subtract the delta to find symmetrical points. We skip the first point
        # since we already set it to mu
        print(sigma, sigma.shape, mu, mu.shape)
        for i in range(0, self.n):
            # REDO THIS TO BE COLUMN centric
            # sigma_points[:, i + 1] = mu.reshape((15,)) + S[:, i - 1]
            # sigma_points[:, i + self.number_of_sigma_points + 1] = (
            #     mu.reshape((15,)) - S[:, i - 1]
            # )
            sigma_points[i + 1, :] = mu + S[i, :].reshape((15, 1))
            sigma_points[i + self.n + 1, :] = mu - S[i, :].reshape((15, 1))

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
        G_q, _ = Rodrigues(orientation)
        R_q = np.linalg.inv(G_q)

        # u_w is the gyroscope measurement, and u_a is the accelerometer
        # measurement from our data point
        xdot[0:3] = velocities
        xdot[3:6] = np.dot(G_q.T, uw.reshape((3, 1)))
        xdot[6:9] = np.array([0, 0, g]).reshape((3, 1)) + np.dot(
            R_q, ua.reshape((3, 1))
        )
        xdot[9:12] = self.ng
        xdot[12:15] = self.na

        # Calculate the new state with our xdot
        return state + (xdot * delta_t)

    def update(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        update takes the current sigma points for the estimate and performs
        the measurement update across them.
        """
        # Use the generated mu and sigma from the predict function (mubar and
        # (sigmabar) to find new sigma points
        sigma_points = self.find_sigma_points(mu, sigma=sigma)

        # Apply the measurement function across each new sigma point
        measurement_points = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            measurement_points[i] = self.measurement_function(sigma_points[i])

        # Calculate the mean of the measurement points by their respective
        # weights. The weights have a 1/N term so the mean is calculated
        # through their addition
        mu = np.zeros((self.n, 1))
        for i in range(0, self.number_of_sigma_points):
            mu += self.weights_mean[i] * measurement_points[i]

        sigma = np.zeros((self.n, self.n))
        differences = measurement_points - mu
        for i in range(0, self.number_of_sigma_points):
            sigma += self.weights_covariance[i] * np.dot(
                differences[i], differences[i].T
            )

    def measurement_function(self, state: np.ndarray) -> np.ndarray:
        """
        measurement_function takes the current state and returns the measurement
        of the state adjusted by our measurement covariance matrix noise

        Note that here state is (6,1), not (15,1), as we are essentially measuring
        6 measurements for the state - the solvePnP positions and orientations.
        """
        noise_adjustment = np.diag(self.measurement_covariance_matrix).reshape(6, 1)
        # Extend it to be a 15,1 to match the state
        noise_adjustment = np.vstack((noise_adjustment, np.zeros((9, 1))))

        return state + noise_adjustment

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
            transitioned_points[i, :] = self.process_model(
                sigma_points[i], delta_t, uw, ua
            )

        # Calculate the mean of the transitioned points by their respective
        # weights. Since we included a 1/N term in the weights, adding them
        # together effectively moves us towards a weighted mean
        Q = 0  # TODO Set Q to something
        mu = np.zeros((self.n, 1))
        for i in range(0, self.number_of_sigma_points):
            mu += (self.weights_mean[i] * transitioned_points[i]) + Q

        # Calculate the covariance of the transitioned points by their
        # respective weights. As before, the weights contain a 1/N term so
        # we are effectively finding the average. We expect a NxN output
        # for our sigma matrix
        differences = transitioned_points - mu
        sigma = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            print("differences sizing", differences[i].shape)
            sigma += self.weights_covariance[i] * np.dot(
                differences[i], differences[i].T
            )
        # Return sigma to 6x6 by just taking the upper left corner
        sigma = sigma[0:6, 0:6]

        return mu, sigma

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
        position, orientation = self.map.estimate_pose(data.tags)
        vector[0:3] = np.array(position).reshape((3, 1))
        vector[3:6] = np.array(orientation_to_yaw_pitch_roll(orientation)).reshape(
            (3, 1)
        )

        # Calculate the velocities/deltas since the last position
        delta_position = vector[0:3] - prior_state[0:3]
        delta_time = data.timestamp - last_timestamp
        vector[6:9] = delta_position / delta_time

        vector[9:12] = self.ng
        vector[12:15] = self.na

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

        for index, data in enumerate(estimated_positions):
            # Skip the 0th estimated position
            if index == 0:
                continue

            # Grab the current state vector for our given estimated position
            state = self.__data_to_vector(data, state, previous_state_timestamp)

            # What is the current accelerometer and gyroscope readings?
            ua = data.acc
            uw = data.rpy

            # Delta t since our last prediction
            delta_t = data.timestamp - previous_state_timestamp
            previous_state_timestamp = data.timestamp

            # Get our sigma points. We expect (self.n * 2) + 1 = 31 sigma points
            # for a (15,31) matrix
            sigma_points = self.find_sigma_points(state)

            # Run the prediction step based off of our state transition
            mu, sigma = self.predict(sigma_points, ua, uw, delta_t)

            # Run the update step to filter our estimated position and resulting
            # sigma (mu and sigma)
            mu, sigma = self.update(mu, sigma)

            # Save the estimated position and timestamp
            pass

        return filtered_positions


if __name__ == "__main__":
    x = UKF()

    dataset = "./hw3/data/studentdata0.mat"

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

    results = x.run(data)
