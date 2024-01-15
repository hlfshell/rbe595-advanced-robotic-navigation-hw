from matplotlib import pyplot as plt
import numpy as np

from typing import Dict, List


# CONSTANTS

OPEN = 1
CLOSED = 0
PUSH = 1
NOOP = 0


class BayesFilter:
    """
    This is a generic implementation of a Bayes Filter, with no
    application-specific implementation details. It simply is
    initiated with a set of prior measurements and initial belief
    of probability. Then you may predict / update as steps to
    iteratively improve the belief of whatever you're trying to
    estimate.
    """

    def __init__(
        self,
        initial_belief: float,
        state_mapping: Dict[int, Dict[int, Dict[int, float]]],
        sensor_probabilities: Dict[int, Dict[int, float]],
    ):
        self.state_mapping = state_mapping
        self.sensor_probabilities = sensor_probabilities
        self.belief = initial_belief
        self.__belief_history: List[float] = [initial_belief]

    def prediction(self, action: int):
        """
        prediction step of the Bayes Filter, returning a new belief
        based on the action taken.
        """
        positive = self.state_mapping[OPEN][action][OPEN] * self.belief
        negative = self.state_mapping[CLOSED][action][OPEN] * (1 - self.belief)
        belief = positive + negative

        return belief

    def update(self, measurement: int):
        """
        update the belief based on the measurement
        """
        is_open = self.sensor_probabilities[measurement][OPEN] * self.belief
        is_closed = self.sensor_probabilities[measurement][CLOSED] * (1 - self.belief)

        normalized_belief = is_open / (is_open + is_closed)

        return normalized_belief

    def calculate_belief(self, action: float, measurement: float) -> float:
        """
        calculate_belief generates a new belief based on the action
        and measurement, calling the prediction and update steps of
        the Bayes filter accordingly.
        """
        self.belief = self.prediction(action)
        self.belief = self.update(measurement)
        self.__belief_history.append(self.belief)
        return self.belief

    def plot_history(self):
        """
        plot_history plots the belief history of the Bayes filter
        """
        plt.plot(self.__belief_history)
        plt.ylabel("State Belief")
        plt.xlabel("Steps")
        plt.title("Bayes Filter Belief History")
        return plt


if __name__ == "__main__":
    # Define our sensor probabilities and state mapping
    # probabilities from actions from the homework

    sensor_probabilities = {
        CLOSED: {
            CLOSED: 0.8,
            OPEN: 0.4,
        },
        OPEN: {
            CLOSED: 0.2,
            OPEN: 0.6,
        },
    }

    state_mapping = {
        CLOSED: {
            PUSH: {
                CLOSED: 0.2,
                OPEN: 0.8,
            },
            NOOP: {
                CLOSED: 1.0,
                OPEN: 0.0,
            },
        },
        OPEN: {
            PUSH: {
                CLOSED: 0.0,
                OPEN: 1.0,
            },
            NOOP: {
                CLOSED: 0.0,
                OPEN: 1.0,
            },
        },
    }

    print(
        "Question 1 - If the robot always takes the action “do nothing” "
        + "and always receives the measurement “door open” how many "
        + "iterations will it take before the robot is at least 99.99% "
        + 'certain the door is open?"'
    )
    initial_belief = 0.5
    filter = BayesFilter(initial_belief, state_mapping, sensor_probabilities)
    action = NOOP
    measurement = OPEN

    iterations = 0
    while filter.belief < 0.9999:
        iterations += 1
        filter.belief = filter.calculate_belief(action, measurement)
    print(f"Reached {filter.belief} in {iterations} iterations.")

    plt = filter.plot_history()
    plt.title("Question 1 - Do Nothing / Measure Open")
    plt.savefig("q1.png")
    plt.show()

    print("")

    print(
        "If the robot always takes the action “push” and always receives "
        + "the measurement “door open” how many iterations will it take "
        + 'before the robot is at least 99.99% certain the door is open?"'
    )
    initial_belief = 0.5
    filter = BayesFilter(initial_belief, state_mapping, sensor_probabilities)
    action = PUSH
    measurement = OPEN

    iterations = 0
    while filter.belief < 0.9999:
        iterations += 1
        filter.belief = filter.calculate_belief(action, measurement)
    print(f"Reached {filter.belief} in {iterations} iterations.")

    plt = filter.plot_history()
    plt.title("Question 2 - Push / Measure Open")
    plt.savefig("q2.png")
    plt.show()

    print("")

    print(
        "If the robot always takes the action “push” and always receives "
        + "the measurement “door closed” what is the steady state belief "
        + "about the door? Include both the state and the certainty."
    )

    # Here we are looking not just for the number of iterations,
    # but essentially a small delta to show that we're at a steady
    # state.
    initial_belief = 0.5
    filter = BayesFilter(initial_belief, state_mapping, sensor_probabilities)
    action = PUSH
    measurement = CLOSED

    minimum_delta = 0.0001
    iterations = 0
    delta = 100.0  # Just a large number to start

    while delta > minimum_delta:
        iterations += 1
        old_belief = filter.belief
        filter.calculate_belief(action, measurement)
        delta = abs(old_belief - filter.belief)

    print(f"Our delta is too small ({delta}), so we are calling it steady now.")
    print(f"Completed in {iterations} iterations.")
    print(f"Final belief: {filter.belief}")

    plt = filter.plot_history()
    plt.title("Question 3 - Push / Measure Closed")
    plt.savefig("q3.png")
    plt.show()
