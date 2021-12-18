from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from lambert import lambert
from nbody import integrate
from utils import get_solar_system
from plot import animate_outcome3d


def earth_to_mars(launch_date, land_date, simulate=False):

    init, G = get_solar_system(["Sun", "Earth", "Mars"], launch_date)
    target, G = get_solar_system(["Sun", "Earth", "Mars"], land_date)

    mu = init[0]["m"] * G

    time_delta = land_date - launch_date
    tof = time_delta.total_seconds()

    launch = np.array(init[1]["p"])
    launch[1] += 0.01
    target = np.array(target[2]["p"])

    v, v_final = lambert(
        mu,
        launch,
        target,
        tof,
        max_steps=10000,
        tol=1e-10,
    )

    if not simulate:
        return v - np.array(init[1]["v"]), v_final
    else:

        init.append(
            {
                "m": init[1]["m"] / 1e6,
                "p": (launch[0], launch[1], launch[2]),
                "v": (v[0], v[1], v[2]),
            }
        )
        outcome, pos = integrate(init, tof, time_delta.days, G)
        return v, v_final, pos


if __name__ == "__main__":
    date = datetime.today()
    land = date + timedelta(days=365)
    v, v_final, pos = earth_to_mars(date, land, simulate=True)

    anim = animate_outcome3d(pos, ["yellow", "blue", "red", "silver"], [30, 20, 15, 5])
    plt.show()
