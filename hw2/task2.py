from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class KalmanFilter:

    def __init__(
        self,
        initial_state: Optional[State],
        measure_velocity: bool,
        mass: float,
        predict_sigma: float,
        update_sigma: float,
    ):
        if initial_state is None:
            initial_state = State((0, 0, 0), (0, 0, 0), 0.0)
        self.state = initial_state
        self.mass = mass
        self.predict_sigma = predict_sigma
        self.update_sigma = update_sigma
        self.P = np.identity(6)

        self.measure_velocity = measure_velocity
        self.H = np.zeros((3, 6))
        if self.measure_velocity:
            self.H[0, 3] = 1
            self.H[1, 4] = 1
            self.H[2, 5] = 1
        else:
            self.H[0, 0] = 1
            self.H[1, 1] = 1
            self.H[2, 2] = 1

    def predict(self, control: Tuple[float, float, float], time: float) -> State:
        """
        Given a control u, update the state of the filter
        """
        F = np.identity(6)
        delta_t = time - self.state.time
        F[0, 3] = delta_t
        F[1, 4] = delta_t
        F[2, 5] = delta_t

        G = np.zeros((6, 3))
        G[0, 0] = delta_t**2 / (self.mass * 2)
        G[1, 1] = delta_t**2 / (self.mass * 2)
        G[2, 2] = delta_t**2 / (self.mass * 2)
        G[3, 0] = delta_t / self.mass
        G[4, 1] = delta_t / self.mass
        G[5, 2] = delta_t / self.mass

        Q = np.identity(6) * (self.predict_sigma**2)

        # Update our P covariance matrix
        self.P = F.dot(self.P).dot(F.T) + Q

        u = np.zeros((3, 1))
        u[0] = control[0]
        u[1] = control[1]
        u[2] = control[2]

        # Produce our estimated state
        x = F.dot(self.__get_state_matrix()) + G.dot(u)

        self.state = State((x[0], x[1], x[2]), (x[3], x[4], x[5]), time)

        return self.state

    def update(self, measurement: Tuple[float, float, float], time: float) -> State:
        """
        Given a measurement z, update the state of the filter
        """
        # Calculate z, our measurement
        x_n = np.zeros((6, 1))
        if self.measure_velocity:
            x_n[3] = measurement[0]
            x_n[4] = measurement[1]
            x_n[5] = measurement[2]
        else:
            x_n[0] = measurement[0]
            x_n[1] = measurement[1]
            x_n[2] = measurement[2]

        z_n = self.H.dot(x_n)

        R = np.identity(3) * (self.update_sigma**2)

        # Calculate the Kalman gain
        K = self.P.dot(self.H.T).dot(
            np.linalg.inv(self.H.dot(self.P).dot(self.H.T) + R)
        )

        # State update equation
        state = self.__get_state_matrix() + K.dot(
            z_n - self.H.dot(self.__get_state_matrix())
        )

        self.state = State(
            (state[0], state[1], state[2]), (state[3], state[4], state[5]), time
        )

        self.P = (np.identity(6) - K.dot(self.H)).dot(self.P).dot(
            np.identity(6) - K.dot(self.H)
        ).T + K.dot(R).dot(K.T)

        return self.state

    def __get_state_matrix(self) -> np.ndarray:
        """
        Return the state matrix
        """
        state_matrix = np.zeros((6, 1))
        state_matrix[0] = self.state.position[0]
        state_matrix[1] = self.state.position[1]
        state_matrix[2] = self.state.position[2]
        state_matrix[3] = self.state.velocity[0]
        state_matrix[4] = self.state.velocity[1]
        state_matrix[5] = self.state.velocity[2]
        return state_matrix


class State(NamedTuple):
    """
    State is a tuple of the position and velocity of the drone
    """

    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    time: float


class DataPoint(NamedTuple):
    """
    DataPoint is a measurement consisting of the timestamp,
    it's measurement, and the system's inputs. Note that for
    this project the measurement may be velocity or position
    depending on which file is read
    """

    t: float
    u1: float
    u2: float
    u3: float
    z1: float
    z2: float
    z3: float


def load_data_from_csv(filepath: str) -> List[DataPoint]:
    with open(filepath, "r") as f:
        lines = f.readlines()
        data = [line.strip().split(",") for line in lines]
        data = [DataPoint(*[float(x) for x in row]) for row in data]
    return data


def generate_plot(data: List[State]) -> plt.Figure:
    """
    Given a list of states, plot the position of the drone
    over time
    """
    # Generate an x_pos, y_pos, and z_pos list
    x_positions = [state.position[0] for state in data]
    y_positions = [state.position[1] for state in data]
    z_positions = [state.position[2] for state in data]

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes(projection="3d")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.dist = 11

    axes.scatter3D(x_positions, y_positions, z_positions, c=z_positions)

    return figure


def run_filter(data: List[DataPoint], filter: KalmanFilter) -> List[DataPoint]:
    """
    Given a specific filter and a a list of data points, run the
    filter step by step over the data points and determine what
    the filter would claim the position of the drone is at each
    point
    """
    result = []
    for i, point in enumerate(data):
        if i == 0:
            filter.predict((point.u1, point.u2, point.u3), point.t)
        else:
            filter.predict((point.u1, point.u2, point.u3), point.t)
            filter.update((point.z1, point.z2, point.z3), point.t)
        result.append(filter.state)
    return result


if __name__ == "__main__":
    # Our drone is 27 grams
    mass = 0.00027

    # Mocap data
    # I'm suggesting a measurement sigma of 1/10th a cm, or 1e-3 m
    # and an update process variance sigma of 0.5 meters
    predict_sigma = 0.5
    update_sigma = 0.001
    data = load_data_from_csv("data/mocap.csv")
    filter = KalmanFilter(None, False, mass, predict_sigma, update_sigma)
    results: List[State] = []
    for datapoint in data:
        filter.predict((datapoint.u1, datapoint.u2, datapoint.u3), datapoint.t)
        filter.update((datapoint.z1, datapoint.z2, datapoint.z3), datapoint.t)
        results.append(filter.state)
    plot = generate_plot(results)
    plot.savefig("imgs/task2_mocap.png")

    # Velocity data
    # I'm suggesting a measurement sigma of 1/10th a cm, or 1e-3 m
    # and an update process variance sigma of 0.5 meters
    predict_sigma = 0.5
    update_sigma = 0.001
    data = load_data_from_csv("data/velocity.csv")
    filter = KalmanFilter(None, True, mass, predict_sigma, update_sigma)
    results: List[State] = []
    for datapoint in data:
        filter.predict((datapoint.u1, datapoint.u2, datapoint.u3), datapoint.t)
        filter.update((datapoint.z1, datapoint.z2, datapoint.z3), datapoint.t)
        results.append(filter.state)
    plot = generate_plot(results)
    plot.savefig("imgs/task2_velocity.png")

    # Low noise
    # Increase our prediction noise from earlier as we have
    # noise to work with
    predict_sigma = 1
    update_sigma = 0.05
    data = load_data_from_csv("data/low_noise.csv")
    filter = KalmanFilter(None, False, mass, predict_sigma, update_sigma)
    results: List[State] = []
    for datapoint in data:
        filter.predict((datapoint.u1, datapoint.u2, datapoint.u3), datapoint.t)
        filter.update((datapoint.z1, datapoint.z2, datapoint.z3), datapoint.t)
        results.append(filter.state)
    plot = generate_plot(results)
    plot.savefig("imgs/task2_low_noise.png")

    # High noise
    # Increase our sigmas for variance even more
    predict_sigma = 1
    update_sigma = 0.1
    data = load_data_from_csv("data/high_noise.csv")
    filter = KalmanFilter(None, False, mass, predict_sigma, update_sigma)
    results: List[State] = []
    for datapoint in data:
        filter.predict((datapoint.u1, datapoint.u2, datapoint.u3), datapoint.t)
        filter.update((datapoint.z1, datapoint.z2, datapoint.z3), datapoint.t)
        results.append(filter.state)
    plot = generate_plot(results)
    plot.savefig("imgs/task2_high_noise.png")
