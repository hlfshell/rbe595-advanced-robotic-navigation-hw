import os
from time import time
from typing import List, Optional, Tuple

import numpy as np
from plot import create_overlay_plots, isometric_plot, plot_rmse_loss
from scipy.linalg import sqrtm
from utils import interpolate_ground_truth, rmse
from world import Coordinate, Data, Map, orientation_to_yaw_pitch_roll, read_mat

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
        Given the state, a delta t, and experienced accelerations from the
        IMU, accelerations from the gyroscope, figure out the resulting
        change of each particle (velocity, etc) and calculate:

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

    def measurement_function(self, state: np.ndarray) -> np.ndarray:
        """
        measurement_function takes the current state and returns the measurement
        of the state adjusted by our measurement covariance matrix noise

        Note that here state is (6,1), not (15,1), as we are essentially measuring
        6 measurements for the state - the solvePnP positions and orientations.
        """
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
        # vector[3:6] = np.array(orientation_to_yaw_pitch_roll(orientation)).reshape(
        #     (3, 1)
        # )
        vector[3:6] = np.array(orientation).reshape((3, 1))

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


if __name__ == "__main__":
    os.makedirs("./hw3/imgs/task4", exist_ok=True)

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

    all_camera_positional_rmses: List[float] = []
    all_ukf_positional_rmses: List[float] = []
    all_camera_orientation_rmses: List[float] = []
    all_ukf_orientation_rmses: List[float] = []

    for dataset in datasets:
        base_data, gt = read_mat(dataset)

        dataset_name = dataset.split("/")[-1].split(".")[0]

        map = Map()
        x = UKF()

        positions: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        times: List[float] = []
        data: List[Data] = []
        interpolated_gt: List[np.ndarray] = []
        camera_estimations: List[np.ndarray] = []
        camera_positional_rmses: List[float] = []
        ukf_positional_rmses: List[float] = []
        camera_orientation_rmses: List[float] = []
        ukf_orientation_rmses: List[float] = []
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
            # orientations.append(orientation_to_yaw_pitch_roll(orientation))
            orientations.append(orientation)
            times.append(datum.timestamp)

            # Calculate the rmse for each point now so we can build a
            # comparison
            position_rmse = rmse(interpolated_gt[-1][0:3], camera_estimations[-1][0:3])
            orientation_rmse = rmse(
                interpolated_gt[-1][3:6], camera_estimations[-1][3:6]
            )
            camera_positional_rmses.append(position_rmse)
            camera_orientation_rmses.append(orientation_rmse)
            all_camera_positional_rmses.append(position_rmse)
            all_camera_orientation_rmses.append(orientation_rmse)

        start = time()
        results = x.run(data)
        print(
            f"Time taken for UKF on dataset {dataset_name}: {time() - start:.2f} seconds"
        )

        # Calculate RMSEs
        for index, result in enumerate(results):
            position_rmse = rmse(interpolated_gt[index][0:3], result[0:3])
            orientation_rmse = rmse(interpolated_gt[index][3:6], result[3:6])
            ukf_positional_rmses.append(position_rmse)
            ukf_orientation_rmses.append(orientation_rmse)
            all_ukf_positional_rmses.append(position_rmse)
            all_ukf_orientation_rmses.append(orientation_rmse)

        # Print out the average camera and UKF rmses
        print(
            f"Average Camera Positional RMSE for {dataset_name}: {np.mean(camera_positional_rmses):.2f}"
        )
        print(
            f"Average UKF Positional RMSE for {dataset_name}: {np.mean(ukf_positional_rmses):.2f}"
        )
        print(
            f"Average Camera Orientation RMSE for {dataset_name}: {np.mean(camera_orientation_rmses):.2f}"
        )
        print(
            f"Average UKF Orientation RMSE for {dataset_name}: {np.mean(ukf_orientation_rmses):.2f}"
        )

        # Create our RMSE plots
        position_rmse, orientation_rmse = plot_rmse_loss(
            interpolated_gt,
            camera_estimations,
            results,
            times,
        )
        position_rmse.savefig(f"./hw3/imgs/task4/{dataset_name}_ukf_position_rmse.png")
        orientation_rmse.savefig(
            f"./hw3/imgs/task4/{dataset_name}_ukf_orientation_rmse.png"
        )

        # Create our overlay plots
        positions_plot, orientations_plot = create_overlay_plots(
            interpolated_gt,
            results,
            times,
        )
        positions_plot.savefig(f"./hw3/imgs/task4/{dataset_name}_ukf_positions.png")
        orientations_plot.savefig(
            f"./hw3/imgs/task4/{dataset_name}_ukf_orientations.png"
        )

        # Create our isometric plots
        isometric = isometric_plot(
            "Isometric View of Ground Truth and Estimated Positions",
            "Ground Truth",
            [Coordinate(x=gt[i].x, y=gt[i].y, z=gt[i].z) for i in range(len(gt))],
            "UKF Estimation",
            [
                Coordinate(x=position[0], y=position[1], z=position[2])
                for position in results
            ],
        )
        isometric.savefig(f"./hw3/imgs/task4/{dataset_name}_ukf_isometric.png")

    # Finally print the mean RMSEs for camera and UKF across all datasets
    print(
        f"Average Camera Positional RMSE for all datasets: {np.mean(all_camera_positional_rmses):.2f}"
    )
    print(
        f"Average UKF Positional RMSE for all datasets: {np.mean(all_ukf_positional_rmses):.2f}"
    )
    print(
        f"Average Camera Orientation RMSE for all datasets: {np.mean(all_camera_orientation_rmses):.2f}"
    )
    print(
        f"Average UKF Orientation RMSE for all datasets: {np.mean(all_ukf_orientation_rmses):.2f}"
    )
