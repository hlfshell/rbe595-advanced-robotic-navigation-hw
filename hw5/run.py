import os

from data import read_trajectory
from plot import create_overlay_plots, isometric_plot, rmse_plots
from ukf import UKF

data = read_trajectory("./hw5/data/trajectory_data.csv")
# data = data[:500]

os.makedirs("./hw5/imgs/", exist_ok=True)
count = 100
for params in [("FB", 2e-3, 5e-5), ("FF", 1e-9, 0.18)]:
    while True:
        count -= 1
        if count < 0:
            print("Failure to complete after 100 runs")
            break

        filter = UKF(
            model_type=params[0],
            noise_scale_measurement=params[1],
            noise_scale_prediction=params[2],
        )
        try:
            estimates, gt, haversines = filter.run(data)
        except:
            continue

        estimates = estimates[1:]
        gt = gt[1:]
        haversines = haversines[1:]
        times = [d.time for d in data][2:]

        isometric_figure = isometric_plot(gt, estimates, data)
        isometric_figure.savefig(f"./hw5/imgs/{params[0]}_isometric.png")

        rmse_figure = rmse_plots(gt, estimates, data)
        rmse_figure.savefig(f"./hw5/imgs/{params[0]}_rmse.png")

        (
            statevar,
            latlon_haversine,
        ) = create_overlay_plots(gt, estimates, haversines, times)
        latlon_haversine.savefig(f"./hw5/imgs/{params[0]}_latlon_haversines.png")
        statevar.savefig(f"./hw5/imgs/{params[0]}_statevar.png")
        break

    if count > 0:
        print(f"Success took {100 - count} runs to complete")
