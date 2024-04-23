from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import sqrtm

from abc import ABC, abstractmethod

from data import Trajectory

from earth import RATE, A, E2, F, gravity, gravity_n, curvature_matrix, principal_radii


class UKF(ABC):

    def __new__(cls) -> UKF:
        return super().__new__()

    def __init__(
        self,
        state_dimensions: int,
        measurement_covariance_matrix: Optional[np.ndarray] = None,
        kappa: float = 1,
        alpha: float = 1,
        beta: float = 2.0,
    ) -> UKF:
        # if measurement_covariance_matrix is None:
        #     self.measurement_covariance_matrix = calculated_covariance_matrix
        # else:
        self.measurement_covariance_matrix = measurement_covariance_matrix

        # n is the number of dimensions for our given state
        self.n = state_dimensions

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

        # Variables we have to keep track of over time
        self.state = np.zeros((self.n, 1))
        self.R_n_bt = np.eye(3)
        self.screw_omega_ne = np.zeros((3, 3))
        self.screw_omega_ei = np.zeros((3, 3))

    @abstractmethod
    def measurement_function(self, state: np.ndarray) -> np.ndarray:
        """
        To be implemented by the subclass. Given a state, return the
        state affected by the measurement function
        """
        pass

    @abstractmethod
    def gyro_acceleration(self):
        """
        ???
        """
        pass

    def find_sigma_points(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        # Based on our system, we expect this to be a 15x31 matrix
        # sigma_points = np.zeros((number_of_points, self.number_of_sigma_points))
        sigma_points = np.zeros((self.number_of_sigma_points, self.n, 1))

        # Set the first column of sigma points to be mu, since that is the mean
        # of our distribution (our center point)
        sigma_points[0] = mu

        S = sqrtm((self.n + self.kappa) * sigma)

        # Now for each point that we wish to go through for our sigma points we
        # move back and forth around the central point; thus we add, then
        # subtract the delta to find symmetrical points. We skip the first point
        # since we already set it to mu
        # This is an implementation of the Julier Sigma Point Method
        for i in range(self.n):
            sigma_points[i + 1] = mu + S[i].reshape((15, 1))
            sigma_points[self.n + i + 1] = mu - S[i].reshape((15, 1))

        return sigma_points

    def process_model(
        self, state: np.ndarray, delta_t: float, ua: np.ndarray, uw=np.ndarray
    ) -> np.ndarray:
        """
        Given a state, delta_t, and existing accelerations, return
        the predicted state after delta_t time has passed.
        """
        # Pull these values from the state
        L = state[0]
        lambda_ = state[1]
        h = state[2]
        phi = state[3]
        theta = state[4]
        psi = state[5]
        Vn = state[6]
        Ve = state[7]
        Vd = state[8]
        e_l = state[9]
        e_lambda = state[10]
        e_h = state[11]

        # Pull these from the prior state
        L_1 = self.state[0]
        lambda_1 = self.state[1]
        h_1 = self.state[2]
        phi_1 = self.state[3]
        theta_1 = self.state[4]
        psi_1 = self.state[5]
        Vn_1 = self.state[6]
        Ve_1 = self.state[7]
        Vd_1 = self.state[8]
        e_l_1 = self.state[9]
        e_lambda_1 = self.state[10]
        e_h_1 = self.state[11]

        #######################
        # Attitude Update
        #######################
        omega_e = RATE
        screw_omega_ei = np.array([[0, -omega_e, 0], [omega_e, 0, 0], [0, 0, 0]])

        R0 = A
        Re_L = R0 / np.sqrt(1 - (E2 * np.sin(L) ** 2))

        omega_ne = np.zeros((3, 1))
        omega_ne[0] = Ve / (Re_L + h)
        omega_ne[1] = -Vn / (Re_L + h)
        omega_ne[2] = -(Ve * np.tan(L)) / (Re_L + h)

        screw_omega_ne = np.array(
            [
                [0, -omega_ne[2], omega_ne[1]],
                [omega_ne[2], 0, -omega_ne[0]],
                [-omega_ne[1], omega_ne[0], 0],
            ]
        )

        screw_omega_bi = np.array(
            [[0, -uw[2], uw[1]], [uw[2], 0, -uw[0]], [-uw[1], uw[0], 0]]
        )

        R_n_bt = self.R_n_bt * (np.eye(3) + screw_omega_bi * delta_t) - (
            (screw_omega_ei + screw_omega_ne) * delta_t * self.R_n_bt
        )

        #######################
        # Velocity Update
        #######################

        f_nt = 1 / 2 * (self.R_n_bt + R_n_bt) * ua
        v_nt = self.v_nt + delta_t * (
            f_nt
            + gravity(L_1, h_1)
            - (Vn_1 * (self.screw_omega_ne + 2 * self.screw_omega_ei))
        )

        #######################
        # Position Update
        #######################
        Rn_1, Re_1, _ = principal_radii(L_1, h_1)
        Rn, Re = principal_radii(L, h)

        h_new = self.h_1 + (delta_t / 2) * (Vd + Vd_1)
        L_new = self.L_1
        L_new += (delta_t / 2) * Vn_1 / (Rn_1 + h_1)
        L_new += (delta_t / 2) * Vn / (Rn_1 + h)

        # Do I

        # Assign stuff to memory for next pass through
        self.R_n_bt = R_n_bt
        self.screw_omega_ne = screw_omega_ne
        self.screw_omega_ei = screw_omega_ei
        self.state = state

    def update(
        self, state: np.ndarray, mu: np.ndarray, sigma: np.ndarray, sigma_points
    ) -> np.ndarray:
        """
        update takes the state, mubar, and sigmabar and performs the
        update step
        """
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

        # Find the cross-covariance
        # Find the differences between the generated sigma points from
        # earlier and mu
        sigmahat_t = np.zeros((self.n, self.n))
        differences_x = sigma_points - mu
        for i in range(0, self.number_of_sigma_points):
            sigmahat_t += self.weights_covariance[i] * np.dot(
                differences_x[i], differences_z[i].T
            )

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

        return mu, sigma, transitioned_points

    def run(self, estimated_positions: List[Trajectory]) -> List[np.ndarray]:
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

        for index, data in enumerate(estimated_positions):
            # Skip the 0th estimated position
            if index == 0:
                continue

            # Grab the current state vector for our given estimated position
            state = self.__data_to_vector(data, state, previous_state_timestamp)

            # What is the current accelerometer and gyroscope readings?
            ua = data.acc
            uw = data.omg

            # Delta t since our last prediction
            delta_t = data.timestamp - previous_state_timestamp
            previous_state_timestamp = data.timestamp

            # Get our sigma points. We expect (self.n * 2) + 1 = 31 sigma points
            # for a (15,31) matrix
            sigma_points = self.find_sigma_points(state, process_covariance_matrix)

            # Run the prediction step based off of our state transition
            mubar, sigmabar, transitioned_points = self.predict(
                sigma_points, ua, uw, delta_t
            )

            # Run the update step to filter our estimated position and resulting
            # sigma (mu and sigma)
            mu, sigma = self.update(state, mubar, sigmabar, transitioned_points)

            # Our current position is mu, and our new process covariance
            # matrix is sigma
            process_covariance_matrix = sigma
            state = mu

            filtered_positions.append(state)

        return filtered_positions
