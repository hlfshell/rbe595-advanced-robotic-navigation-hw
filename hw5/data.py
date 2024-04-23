from typing import List, NamedTuple


class Trajectory(NamedTuple):
    time: float
    true_lat: float
    true_lon: float
    true_alt: float
    true_roll: float
    true_pitch: float
    true_heading: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    accel_x: float
    accel_y: float
    accel_z: float
    z_lat: float
    z_lon: float
    z_alt: float
    z_VN: float
    z_VE: float
    z_VD: float


def read_trajectory(filename: str, headings_included: bool = True) -> List[Trajectory]:
    """
    Read the given CSV file and return the Trajectory points provided
    """
    trajectories = []
    with open(filename) as f:
        for line in f:
            parts = line.split(",")
            # Skip the headings if headings are included)
            if headings_included:
                headings_included = False
                continue
            trajectories.append(
                Trajectory(
                    time=float(parts[0]),
                    true_lat=float(parts[1]),
                    true_lon=float(parts[2]),
                    true_alt=float(parts[3]),
                    true_roll=float(parts[4]),
                    true_pitch=float(parts[5]),
                    true_heading=float(parts[6]),
                    gyro_x=float(parts[7]),
                    gyro_y=float(parts[8]),
                    gyro_z=float(parts[9]),
                    accel_x=float(parts[10]),
                    accel_y=float(parts[11]),
                    accel_z=float(parts[12]),
                    z_lat=float(parts[13]),
                    z_lon=float(parts[14]),
                    z_alt=float(parts[15]),
                    z_VN=float(parts[16]),
                    z_VE=float(parts[17]),
                    z_VD=float(parts[18]),
                )
            )

    return trajectories
