from data import read_trajectory
from ukf import UKF

import numpy as np
from plot import rmse

data = read_trajectory("./hw5/data/trajectory_data.csv")
data = data[:500]

# measurements = [5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 1]
measurements = np.arange(1, 15, 0.1)
measurements = []
predictions = []

for i in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    for j in np.arange(0.0, 1.0, 0.1):
        predictions.append(i + i * j)
        measurements.append(i + i * j)

successes = []

failures = []

count = 0
for measurement_noise in measurements:
    for prediction_noise in predictions:
        count += 1
        print(
            f"{count}/{len(measurements) * len(predictions)} - Failures: {len(failures)} Successes: {len(successes)}",
            end="\r",
        )
        try:
            filter = UKF(
                model_type="FF",
                noise_scale_measurement=measurement_noise,
                noise_scale_prediction=prediction_noise,
            )
            estimates, gt, haversines = filter.run(data)
        except:
            failures.append((measurement_noise, prediction_noise, filter.steps))
            continue

        # Now we measure the random rmse of each estimate against its associated
        # ground truth
        errors = []
        for index, estimate in enumerate(estimates):
            truth = gt[index + 1][0:2]
            est = estimate[0:2]
            errors.append(rmse(truth, est))
        # Calc the mean of the errors
        mean_error = np.mean(errors)
        mean_haversines = np.mean(haversines)

        # print(
        #     "Success", measurement_noise, prediction_noise, mean_error, mean_haversines
        # )
        successes.append(
            (measurement_noise, prediction_noise, mean_error, mean_haversines)
        )

print("Top 10 failures")
# Sort the failures based on the steps count-  higher is better
failures.sort(key=lambda x: x[2], reverse=True)
for index, failure in enumerate(failures[:10]):
    print(
        f"{index+1}. Measurement Noise: {failure[0]}, Prediction Noise: {failure[1]}, Steps: {failure[2]}"
    )

if len(successes) == 0:
    print("No successes :*(")
else:
    print("Successes in order:")
    # Sort the successes based on lowest mean_error
    successes.sort(key=lambda x: x[2])
    for success in successes:
        print(
            f"Measurement Noise: {success[0]}, Prediction Noise: {success[1]}, Average RMSE: {success[2]}"
        )
