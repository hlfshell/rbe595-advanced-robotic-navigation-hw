from __future__ import annotations

from typing import List, Tuple

import numpy as np
from data import Data
from earth import RATE, gravity_n, principal_radii
from haversine import Unit, haversine
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation


class UKF:

    def __init__(
        self,
        kappa: float = 1.0,
        alpha: float = 1.0,
        beta: float = 0.4,
        model_type: str = "FB",
        noise_scale_measurement: float = 5e-3,
        noise_scale_prediction: float = 1e-3,
    ) -> UKF:
        self.model_type = model_type

        # n is the number of dimensions for our given state
        state_dimensions = 12 if self.model_type == "FF" else 15
        self.n = state_dimensions

        self.noise_scale_measurement = noise_scale_measurement
        self.noise_scale_prediction = noise_scale_prediction

        # The number of sigma points we do are typically 2*n + 1,
        # so therefore...
        self.number_of_sigma_points = 2 * self.n + 1  # 25 or 31 based on model
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

    def measurement_function(self, state: np.ndarray, gnss: np.ndarray) -> np.ndarray:
        """
        measurement_function takes the current state and returns the measurement
        of the state adjusted by our measurement covariance matrix noise

        Note that here state is (6,1), not (15,1), as we are essentially measuring
        6 measurements for the state - the solvePnP positions and orientations.
        """
        noise_adjustment = np.zeros((self.n, 1))
        if self.model_type == "FF":
            noise_scale = 5e-3
        else:
            noise_scale = 2e-3
        noise = np.random.normal(
            scale=self.noise_scale_measurement, size=(self.n, self.n)
        )
        # noise = np.random.normal(scale=noise_scale, size=(self.n, self.n))
        c = np.zeros((self.n, self.n))
        c[0:6, 0:6] = np.eye(6)
        if self.model_type == "FB":
            c[6:9, 6:9] = np.eye(3)

        R = np.diag(noise).reshape(self.n, 1)
        noise_adjustment[0 : self.n] = np.dot(c, state) + R

        # Calculate error difference to measurement
        # if self.model_type == "FF":
        #     haversine_distance = haversine(state[0:2], gnss[0:2], unit=Unit.DEGREES)
        #     noise_adjustment[9] = haversine_distance
        #     noise_adjustment[10] = haversine_distance
        #     noise_adjustment[11] = state[2] - gnss[2]

        return noise_adjustment

    def find_sigma_points(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        # Based on our system, we expect this to be a 15x31 matrix
        sigma_points = np.zeros((self.number_of_sigma_points, self.n, 1))

        # Set the first column of sigma points to be mu, since that is the mean
        # of our distribution (our center point)
        sigma_points[0] = mu

        try:
            S = sqrtm((self.n + self.kappa) * sigma)
        except:
            print(self.n)
            print(self.kappa)
            print(sigma)
            print(sigma.shape)
            raise "sqrtm failure"

        # Now for each point that we wish to go through for our sigma points we
        # move back and forth around the central point; thus we add, then
        # subtract the delta to find symmetrical points. We skip the first point
        # since we already set it to mu
        # This is an implementation of the Julier Sigma Point Method
        for i in range(self.n):
            sigma_points[i + 1] = mu + S[i].reshape((self.n, 1))
            sigma_points[self.n + i + 1] = mu - S[i].reshape((self.n, 1))

        return sigma_points

    def propagation_model(
        self, state: np.ndarray, delta_t: float, fb: np.ndarray, wb=np.ndarray
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

        v_n = np.array([Vn, Ve, Vd]).reshape(3, 1)

        if self.model_type == "FB":
            fb -= state[9:12]
            wb -= state[12:15]

        R_nb_prev = Rotation.from_euler(
            "xyz", np.array([phi, theta, psi]).reshape((3,)), degrees=True
        ).as_matrix()

        #######################
        # Attitude Update
        #######################
        omega_e = RATE
        screw_omega_ei = np.array([[0, -omega_e, 0], [omega_e, 0, 0], [0, 0, 0]])

        Rn_Lh, Re_Lh, Re_LhcosL = principal_radii(L, h)

        omega_ne = np.zeros((3,))
        omega_ne[0] = Ve / Re_Lh
        omega_ne[1] = -Vn / Rn_Lh
        omega_ne[2] = -(Ve * np.tan(np.deg2rad(L))) / Re_Lh

        screw_omega_ne = np.array(
            [
                [0, -omega_ne[2], omega_ne[1]],
                [omega_ne[2], 0, -omega_ne[0]],
                [-omega_ne[1], omega_ne[0], 0],
            ]
        )

        omega_ne = omega_ne.reshape((3, 1))

        wb = wb.reshape((3,))
        screw_omega_bi = np.array(
            [[0, -wb[2], wb[1]], [wb[2], 0, -wb[0]], [-wb[1], wb[0], 0]]
        )
        wb = wb.reshape((3, 1))

        R_nb = R_nb_prev * (np.eye(3) + screw_omega_bi * delta_t) - (
            (screw_omega_ei + screw_omega_ne) * delta_t * R_nb_prev
        )

        #######################
        # Velocity Update
        #######################

        f_nt = 1 / 2 * np.dot(R_nb_prev + R_nb, fb)

        v_nt = v_n + delta_t * (
            f_nt
            + gravity_n(L, h).reshape((3, 1))
            - np.dot(screw_omega_ne + 2 * screw_omega_ei, v_n)
        )

        #######################
        # Position Update
        #######################
        h_new = h - (delta_t / 2) * (Vd + v_nt[2])
        Rn_Lhnew, _, _ = principal_radii(L, h_new)
        L_new = L
        L_new += (delta_t / 2) * (Vn / Rn_Lh + v_nt[0] / Rn_Lh)
        L_new += (delta_t / 2) * (Vn / Rn_Lh + v_nt[0] / Rn_Lhnew)

        _, _, Re_LhcosLnew = principal_radii(L_new, h_new)

        lambda_new = lambda_
        lambda_new += (delta_t / 2) * (Ve / Re_LhcosL)
        lambda_new += (delta_t / 2) * (Ve / Re_LhcosLnew)

        phi, theta, psi = Rotation.as_euler(
            Rotation.from_matrix(R_nb),
            "xyz",
            degrees=True,
        )

        new_state = np.zeros((self.n, 1))
        new_state[0] = L_new
        new_state[1] = lambda_new
        new_state[2] = h_new
        new_state[3] = phi
        new_state[4] = theta
        new_state[5] = psi
        new_state[6:9] = v_nt.reshape((3, 1))
        # We aren't modifying the biases, so keep
        # them as they were passed in
        new_state[9:] = state[9:]

        return new_state

    def update(
        self,
        gnss: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        sigma_points: np.ndarray,
    ) -> np.ndarray:
        """
        update takes the gnss, mubar, and sigmabar and performs the
        update step
        """
        # Apply the measurement function across each new sigma point
        measurement_points = np.zeros_like(sigma_points)
        for i in range(self.number_of_sigma_points):
            measurement_points[i] = self.measurement_function(sigma_points[i], gnss)

        # Calculate the mean of the measurement points by their respective
        # weights. The weights have a 1/N term so the mean is calculated
        # through their addition
        zhat = np.zeros((self.n, 1))
        for i in range(0, self.number_of_sigma_points):
            zhat += self.weights_mean[i] * measurement_points[i]

        St = np.zeros((self.n, self.n))
        differences_z = measurement_points - zhat
        for i in range(0, self.number_of_sigma_points):
            St += self.weights_covariance[i] * np.dot(
                differences_z[i], differences_z[i].T
            )
        # St += R

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
        current_position = mu + np.dot(kalman_gain, gnss - zhat)
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
        self, sigma_points: np.ndarray, fb: np.ndarray, wb: np.ndarray, delta_t: float
    ) -> Tuple[np.ndarray]:
        """
        predict takes the current sigma points for the estimate and performs
        the state transition across them. We then compute the mean and the
        covariance of the resulting transformed sigma points.
        """
        # For each sigma point, run them through our state transition function
        transitioned_points = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            transitioned_points[i, :] = self.propagation_model(
                sigma_points[i], delta_t, wb, fb
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
        if self.model_type == "FF":
            noise_scale = 1e-3
        else:
            noise_scale = 5e-5
        Q = np.random.normal(scale=self.noise_scale_prediction, size=(self.n, self.n))
        # Q = np.random.normal(scale=noise_scale, size=(self.n, self.n))
        differences = transitioned_points - mu
        sigma = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            sigma += self.weights_covariance[i] * np.dot(
                differences[i], differences[i].T
            )
        sigma += Q

        return mu, sigma, transitioned_points

    def imu_from_data(self, data: Data) -> np.ndarray:
        d = np.zeros((6, 1))
        d[0] = data.accel_x
        d[1] = data.accel_y
        d[2] = data.accel_z
        d[3] = data.gyro_x
        d[4] = data.gyro_y
        d[5] = data.gyro_z

        return d

    def gnss_from_data(self, data: Data) -> np.ndarray:
        z = np.zeros((self.n, 1))
        z[0] = data.z_lat
        z[1] = data.z_lon
        z[2] = data.z_alt
        z[6] = data.z_VN
        z[7] = data.z_VE
        z[8] = data.z_VD

        return z

    def true_pose_from_data(self, data: Data) -> np.ndarray:
        state = np.zeros((self.n, 1))
        state[0] = data.true_lat
        state[1] = data.true_lon
        state[2] = data.true_alt
        state[3] = data.true_roll
        state[4] = data.true_pitch
        state[5] = data.true_heading

        return state

    def run(
        self, data: List[Data]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Given a set of estimated positions, return the estimated positions after running
        the Unscented Kalman Filter over the data.

        Returns:
        1. The filter's estimated states (n - 1 states)
        2. The ground truth positions (n states)
        3. The haversine distances between the estimated and ground truth positions (n-1 distances)
        """
        ground_truth: List[np.ndarray] = [self.true_pose_from_data(d) for d in data]
        filtered_positions: List[np.ndarray] = []
        haversine_distances: List[np.ndarray] = []

        # First we need to initialize our initial position to the 0th estimated
        # position. We will use the true value for the init on the 0th only
        state = self.true_pose_from_data(data[0])
        previous_state_timestamp = data[0].time

        process_covariance_matrix = np.eye(self.n) * 1e-3
        self.steps = 0

        for index, read in enumerate(data):
            # Skip the 0th estimated position
            if index == 0:
                continue

            self.steps += 1
            # Delta t since our last prediction
            delta_t = read.time - previous_state_timestamp
            previous_state_timestamp = read.time
            imu = self.imu_from_data(read)
            gnss = self.gnss_from_data(read)
            fb = imu[0:3]
            wb = imu[3:6]

            # Get our sigma points. We expect (self.n * 2) + 1 for S sigma points
            # for a (self.n, S) matrix
            sigma_points = self.find_sigma_points(state, process_covariance_matrix)

            # Run the prediction step based off of our state transition
            mubar, sigmabar, transitioned_points = self.predict(
                sigma_points, fb, wb, delta_t
            )

            # Handle error calculation if the model is feed forward
            if self.model_type == "FF":
                mubar

            # Run the update step to filter our estimated position and resulting
            # sigma (mu and sigma)
            mu, sigma = self.update(gnss, mubar, sigmabar, transitioned_points)

            # Our current position is mu, and our new process covariance
            # matrix is sigma
            process_covariance_matrix = sigma
            state = mu

            if self.model_type == "FF":
                mubar[9:12] = mubar[0:3] - gnss[0:3]

            filtered_positions.append(state)
            haversine_distances.append(
                haversine(
                    (state[0], state[1]),
                    (ground_truth[index][0], ground_truth[index][1]),
                    unit=Unit.DEGREES,
                )
            )

        return filtered_positions, ground_truth, haversine_distances
