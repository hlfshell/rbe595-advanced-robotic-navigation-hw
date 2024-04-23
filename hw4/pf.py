from typing import List, Tuple

import numpy as np
from covariance import calculated_covariance_matrix
from world import Data, Map, orientation_to_yaw_pitch_roll


class ParticleFilter:

    def __init__(
        self,
        particle_count=2_000,
        covariance_matrix: np.ndarray = calculated_covariance_matrix,
        minimum_effective_particle_count=0.5,
        noise_scale: float = 105,
        noise_scale_gyro: float = 0.01,
    ):
        self.particle_count = particle_count
        self.covariance_matrix = covariance_matrix
        self.minimum_effective_particle_count = (
            minimum_effective_particle_count * particle_count
        )
        self.map = Map()
        self.noise_scale = noise_scale
        self.noise_scale_gyro = noise_scale_gyro

    def update(self, measured_state: np.ndarray) -> np.ndarray:
        """
        Takes the measured state and performs a predict step based off of
        the measurement and our available covariance matrix
        """
        # First create our measurement matrix, which is to be a 6x15
        # matrix based off our measurement and state vector
        C = np.zeros((6, 15))
        # The upper left hand corner of our measurement matrix is I
        # For our estimated position and orientation, 0's for IMU
        C[0:6, 0:6] = np.identity(6)

        # Grab the diagonal of our covariance matrix
        covariance_diagonal = np.diag(self.covariance_matrix).reshape((6, 1))

        return np.dot(C, measured_state) + covariance_diagonal

    def effective_number_of_particles(self, weights: np.ndarray) -> float:
        """
        Given a set of weights, calculate the effective number of particles,
        wherein effective particles are the number of particles contributing
        meaningfully to the filter.
        """
        return 1 / np.sum(weights**2)

    def resample(
        self, particles: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a set of particles and their weights, resample the particles
        with a low variance approach (also called stratified resampling).

        Returns a new set of particles and weights, respectively
        """
        resampled_particles = np.zeros((self.particle_count, 15, 1))
        resampled_weights = np.zeros((self.particle_count, 1))

        # Initialize state variables for our resampling
        cumulative_sum = weights[0]  # Set to our first weight
        index = 0

        # Generate our random starting point, a value somewhere in
        # the interval of (0, 1/particle_count).
        r = np.random.uniform(0, 1 / self.particle_count)

        for i in range(self.particle_count):
            # Calculate our next sample point
            sample_point = r + (i * (1 / self.particle_count))

            # Find the first weight that is beyond the generated
            # sample point
            while cumulative_sum < sample_point:
                index += 1
                cumulative_sum += weights[index]

            # Select the particle at the current index and copy
            # its weight over as well
            resampled_particles[i] = particles[index]
            resampled_weights[i] = weights[index]

        # Now that we have the weights, we need to renormalize them
        # to add up to 1
        resampled_weights = resampled_weights / np.sum(resampled_weights)

        return resampled_particles, resampled_weights

    def measurement(self, data: Data) -> np.ndarray:
        """
        Given a data point, return a measurement matrix of the data
        for working with
        """
        if not data.tags:
            raise ValueError("No AprilTags present on data point")

        orientation, position = self.map.estimate_pose(data.tags)
        # orientation = orientation_to_yaw_pitch_roll(orientation)

        state = np.zeros((15, 1))
        state[0:3] = position.reshape((3, 1))
        state[3:6] = np.array(orientation).reshape((3, 1))

        return state

    def predict(
        self,
        particles: np.ndarray,
        delta_t: float,
        accelerometer: np.ndarray,
        gyroscope: np.ndarray,
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

        # We create a noise vector to add to our particles at
        # the end of the state transition process
        noise = np.zeros((self.particle_count, 6, 1))
        # We do individual noise vectors for each particle because
        # we want to tune the range of noise possible for the
        # given vector. We only do noise across the 0-6 indicies as
        # that's the measured noise we're aiming to add onto (IMU)
        noise[:, 0:3] = np.random.normal(
            scale=self.noise_scale, size=(self.particle_count, 3, 1)
        )
        noise[:, 3:6] = np.random.normal(
            scale=self.noise_scale_gyro, size=(self.particle_count, 3, 1)
        )

        xdot = np.zeros((self.particle_count, 15, 1))
        # ua and uw are the bias from the accelerometer and gyroscope
        # respectively.
        ua = np.tile(accelerometer.reshape((3, 1)), (self.particle_count, 1, 1))
        uw = np.tile(gyroscope.reshape(3, 1), (self.particle_count, 1, 1))
        g = -9.81

        # # Add our noise to our ua/uw
        ua = ua + noise[:, 0:3]
        # uw = uw + noise[:, 3:6]

        # Extract the orientation, and velocities from the state
        orientations = particles[:, 3:6]
        velocities = particles[:, 6:9]

        # Create our rotation matrix from the drone frame to the world frame
        # via our orientation roll pitch yaw. We do this by finding G_q and
        # then solving G_q's inverse. The inverse is a when doing it
        # vectorized w/ all particles at once; so we quickly just manually
        # use the analytical result off the bat.
        thetas = orientations[:, 0]
        phis = orientations[:, 1]
        psis = orientations[:, 2]

        G_q_inv = np.zeros((self.particle_count, 3, 3, 1))
        G_q_inv[:, 0, 0] = np.cos(thetas)
        # G_q_inv[:, 0, 1] = 0.0
        G_q_inv[:, 0, 2] = np.sin(thetas)
        G_q_inv[:, 1, 0] = np.sin(phis) * np.sin(thetas) / np.cos(phis)
        G_q_inv[:, 1, 1] = 1.0
        G_q_inv[:, 1, 2] = -np.cos(thetas) * np.sin(phis) / np.cos(phis)
        G_q_inv[:, 2, 0] = -np.sin(thetas) / np.cos(phis)
        # G_q_inv[:, 2, 1] = 0.0
        G_q_inv[:, 2, 2] = np.cos(thetas) / np.cos(phis)

        # We create R_q, again manually, to keep things a bit easier
        R_q = np.zeros((self.particle_count, 3, 3, 1))
        R_q[:, 0, 0] = np.cos(psis) * np.cos(thetas) - np.sin(phis) * np.sin(
            phis
        ) * np.sin(thetas)
        R_q[:, 0, 1] = -np.cos(phis) * np.sin(psis)
        R_q[:, 0, 2] = np.cos(psis) * np.sin(thetas) + np.cos(thetas) * np.sin(
            phis
        ) * np.sin(psis)
        R_q[:, 1, 0] = np.cos(thetas) * np.sin(psis) + np.cos(psis) * np.sin(
            phis
        ) * np.sin(thetas)
        R_q[:, 1, 1] = np.cos(phis) * np.cos(psis)
        R_q[:, 1, 2] = np.sin(psis) * np.sin(thetas) - np.cos(psis) * np.cos(
            thetas
        ) * np.sin(phis)
        R_q[:, 2, 0] = -np.cos(phis) * np.sin(thetas)
        R_q[:, 2, 1] = np.sin(phis)
        R_q[:, 2, 2] = np.cos(phis) * np.cos(thetas)

        xdot[:, 0:3] = velocities

        # Iterate through each particle because I couldn't get it to
        # work in full vector form.
        for i in range(self.particle_count):
            xdot[i, 3:6] = np.dot(G_q_inv[i].reshape((3, 3)), uw[i].reshape((3, 1)))
            xdot[i, 6:9] = np.array([0, 0, g]).reshape((3, 1)) + np.dot(
                R_q[i].reshape((3, 3)), ua[i].reshape((3, 1))
            )

        # We are ignoring this for now; it's defaulted to zero so no need
        # to do the zeroing again.
        # xdot[:, 9:12] = np.zeros((3, 1))
        # xdot[:, 12:15] = np.zeros((3, 1))

        # Add our xdot delta to our particles
        # particles = particles + (xdot * delta_t)
        xdot = xdot * delta_t
        # Add noise again for gyro
        xdot[:, 3:6] = xdot[:, 3:6] + noise[:, 3:6]
        particles = particles + xdot

        return particles

    def update_weights(
        self, particles: np.ndarray, prediction: np.ndarray
    ) -> np.ndarray:
        """
        Given a prediction, update the weights of the particles based on
        the calculated prediction.
        """
        # Calculate our error, which is the difference between the particle's
        # current state and the prediction. prediction is (15,1) and particles
        # is (particle_count, 15, 1). Despite the size differences, the
        # subtraction works as you'd expect. We only focus on the 0:6 indicies
        # as that's our primary focus for the particle filter.
        errors = particles[:, 0:6] - prediction[0:6]

        # The weights will be:
        #           1
        #   ---------------------
        #   sqrt(Sigma(errors^2))

        weights = np.exp(-0.5 * np.sum(errors**2, axis=1))

        # Normalize the weights, so we only operate with weights that
        # add to 1
        weights = weights / np.sum(weights)

        return weights

    def weighted_average(
        self, particles: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Given the particles and their weights, calculate a singular point
        from their weighted average
        """
        # We need to reshape the weights to a (particle_count, 1, 1) to
        # make the multiplication work the way we'd expect across each
        # particle.
        weighted_particles = particles * weights.reshape(self.particle_count, 1, 1)
        summed_particles = np.sum(weighted_particles, axis=0)

        return summed_particles / np.sum(weights)

    def highest_particle(
        self, particles: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Given the particles and their weights, return the particle with
        the highest weight.
        """
        return particles[np.argmax(weights)]

    def average(self, particles: np.ndarray) -> np.ndarray:
        """
        Given a set of particles, return the average of all particles
        """
        return np.sum(particles, axis=0) / self.particle_count

    def initial_particles(self) -> np.ndarray:
        """
        Given what we know of the problem, create a uniform distribution
        of particles for initial filtering.
        """
        # These ranges are the range of possible coordinates based on
        # where we can find the drone in the world. It's grabbed from
        # observation that we're not going to find the drone outside of
        # these ranges.
        x_range = (0.0, 3)
        y_range = (0.0, 3)
        z_range = (0, 1.5)  # We never see the drone higher than ~1.2 meters

        # Orientation ranges are not full circle, as we never really
        # observe the drone upside down, so we do limit the estimates
        # of orientation.
        yaw_range = (-0.5 * np.pi, 0.5 * np.pi)
        pitch_range = (-0.5 * np.pi, 0.5 * np.pi)
        roll_range = (-0.5 * np.pi, 0.5 * np.pi)

        # Lows and Highs
        lows = [
            x_range[0],
            y_range[0],
            z_range[0],
            yaw_range[0],
            pitch_range[0],
            roll_range[0],
        ]
        highs = [
            x_range[1],
            y_range[1],
            z_range[1],
            yaw_range[1],
            pitch_range[1],
            roll_range[1],
        ]

        # Uniform takes low, high, and size. It will do the [low, high] for
        # each column of the given size, wherein we need it by row. So we
        # will create this at our desired size and transpose it.
        particles = np.random.uniform(
            low=lows, high=highs, size=(self.particle_count, 6)
        )

        # Even though we are only filling the first 6 rows, we want
        # this to be a (15,1) to match our state vector, so we will
        # concatenate zeros accordingly
        particles = np.concatenate(
            (particles, np.zeros((self.particle_count, 9))), axis=1
        )

        # We are expecting the result to be particle count of 6x1s, so we
        # will expand the dimensions to match that.
        particles = np.expand_dims(particles, axis=-1)

        return particles

    def run(
        self, estimated_positions: List[Data], estimate_method: str = "weighted"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        run will take a set of points and perform the particle filter
        over each, zeroing in on an ultimate filter estimated state
        at each presented timestep.

        The function returns a tuple of estimates, particles, where
        estimates is the resulting estimated positions at each timestep,
        and particles is the state of all particles being utilized at
        each timestep.

        The estimate_method specifies how the estimate is derived in each
        iteration:
            - weighted - a weighted average of all particles based on
                    their weights
            - average - the average of all particles ignoring weights
            - highest - the particle with the highest weight
        """
        if estimate_method not in ["weighted", "average", "highest"]:
            raise "Invalid estimate method"

        # First confirm that we have filtered out positions without AprilTags
        # as we cannot estimate position without them.
        for index, position in enumerate(estimated_positions):
            if not position.tags:
                raise ValueError(f"No AprilTags present on data point at index {index}")

        # Create final results arrays for both the predicted positions and
        # per step particles
        estimates = np.zeros((len(estimated_positions) - 1, 15, 1))
        particle_history = np.zeros(
            (len(estimated_positions), self.particle_count, 15, 1)
        )

        # We start with an initialized set of particles, set time to 0.0,
        # and initialize our position as our first known position
        particles = self.initial_particles()
        particle_history[0] = particles
        time = estimated_positions[0].timestamp

        # Weights are defaulted to 1/N, or equal for each particle
        weights = np.ones((self.particle_count, 1)) / self.particle_count

        for index, position in enumerate(estimated_positions):
            # Skip the first position with april tags
            if index == 0:
                continue
            # if index == 10:
            #     raise "die"
            # print(index, end=" ")
            # Print a progress indicator that doesn't use a new line,
            # clears the text and then prints the current index again
            print(f"\r{index+1}/{len(estimated_positions)}", end="")

            delta_t = position.timestamp - time
            time = position.timestamp

            # Perform the state transition across all particles
            particles = self.predict(particles, delta_t, position.acc, position.omg)
            particle_history[index] = particles

            # Get our current measurement of our estimated position
            measurement = self.measurement(position)

            # Take the measurement and perform a prediction step off
            # of it.
            prediction = self.update(measurement)

            # Expand prediction from the 6x1 return to our 15x1 shape
            # (default to 0's on all other values)
            prediction = np.concatenate((prediction, np.zeros((9, 1))))

            # Update our weights based on the prediction
            # print("weights sum prior to update_weights", np.sum(weights))
            weights = self.update_weights(particles, prediction)
            # print("weights sum post update_weights", np.sum(weights))

            # Add the estimate wherein I use the weights to create an estimate
            if estimate_method == "weighted":
                estimate = self.weighted_average(particles, weights)
            elif estimate_method == "average":
                estimate = self.average(particles)
            elif estimate_method == "highest":
                estimate = self.highest_particle(particles, weights)

            # print(
            #     f"{index} - {[f'{e[0]:.3f}' for e in estimate[0:3]]} {[f'{e[0]:.3f}' for e in estimate[3:6]]}"
            # )

            # Resample the particles
            particles, weights = self.resample(particles, weights)

            estimates[index - 1] = estimate

        return estimates, particle_history