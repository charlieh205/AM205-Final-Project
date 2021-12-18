from earth_to_mars import earth_to_mars
from astropy import units as u
from poliastro.util import norm
import matplotlib.pyplot as plt
import numpy as np


class Porkchop(object):
    def __init__(
        self, launch_span, arrival_span, max_c3=45.0 * u.km ** 2 / u.s ** 2, ax=None
    ):
        self.launch_span = launch_span
        self.arrival_span = arrival_span
        self.max_c3 = max_c3
        self.ax = ax
        self.launch_dates = [str(x).split(" ")[0] for x in launch_span]
        self.arrival_dates = [str(x).split(" ")[0] for x in arrival_span]

    def _get_c3_launch(self):
        c3 = 1000 * np.ones((len(self.launch_span), len(self.arrival_span)))
        for i in range(len(self.launch_span)):
            launch = self.launch_span[i].to_datetime()
            print(f"{i+1}/{len(self.launch_span)}")
            for j in range(len(self.arrival_span)):
                arrival = self.arrival_span[j].to_datetime()
                duration_days = (arrival - launch).days
                if duration_days > 30:
                    dv_dpt, _ = earth_to_mars(launch, arrival)
                    dv_dpt *= 1.496e8
                    dv_dpt = dv_dpt * (u.km / u.s)
                    dv_dpt = dv_dpt.to(u.m / u.s)
                    dv_dpt = norm(dv_dpt)
                    c3_launch = dv_dpt ** 2
                    c3[j, i] = c3_launch.to_value(u.km ** 2 / u.s ** 2)
        return c3

    def plotter(self, filename, num_best=10, plot_best=False):
        c3_launch = self._get_c3_launch()

        if self.ax is None:
            fig, self.ax = plt.subplots(figsize=(15, 15))
        else:
            fig = self.ax.figure

        c3_levels = np.linspace(0, self.max_c3.to_value(u.km ** 2 / u.s ** 2), 30)
        c = self.ax.contourf(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            c3_launch,
            c3_levels,
        )
        line = self.ax.contour(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            c3_launch,
            c3_levels,
            colors="black",
            linestyles="solid",
        )
        cbar = fig.colorbar(c)
        cbar.set_label("km2 / s2")
        self.ax.clabel(line, inline=1, fmt="%1.1f", colors="k", fontsize=10)
        self.ax.grid()
        fig.autofmt_xdate()
        self.ax.set_title(
            f"Earth - Mars for year {self.launch_span[0].datetime.year}, C3 Launch",
            fontsize=14,
            fontweight="bold",
        )
        self.ax.set_xlabel("Launch date", fontsize=10, fontweight="bold")
        self.ax.set_ylabel("Arrival date", fontsize=10, fontweight="bold")

        c3_launch = c3_launch * u.km ** 2 / u.s ** 2

        (l, a) = np.unravel_index(
            np.argmin(np.transpose(c3_launch), axis=None), c3_launch.shape
        )

        best_launch = self.launch_dates[l]
        best_arrival = self.arrival_dates[a]

        if plot_best:
            lab = f"Launch date: {best_launch}\nArrival date: {best_arrival}"
            self.ax.scatter([best_launch], [best_arrival], color="red", label=lab)
            self.ax.legend(loc="best")

        plt.savefig(f"ours_{filename}.svg", bbox_inches="tight")
        return best_launch, best_arrival
