from earth_to_mars import earth_to_mars
from astropy import units as u
from poliastro.util import norm
import matplotlib.pyplot as plt
import numpy as np

class Porkchop(object):

    def __init__(
        self,
        launch_span,
        arrival_span,
        max_c3 = 1000.0  * u.km**2 / u.s**2,
        ax = None
    ):
        self.launch_span = launch_span
        self.arrival_span = arrival_span
        self.max_c3 = max_c3
        self.ax = ax
        self.launch_dates = [str(x).split(' ')[0] for x in launch_span]
        self.arrival_dates = [str(x).split(' ')[0] for x in arrival_span]

    def _get_c3_launch(self):
        c3 = np.zeros((len(self.launch_span), len(self.arrival_span)))
        for i in range(len(self.launch_span)):
            launch = self.launch_span[i].to_datetime()
            for j in range(len(self.arrival_span)):
                arrival = self.arrival_span[j].to_datetime()
                dv_dpt, _ = earth_to_mars(launch, arrival)
                dv_dpt *= 1.496e8
                dv_dpt = dv_dpt * (u.km / u.s)
                # if i == 0 and j == 0:
                #     print(f"Launch: {launch}, arrival: {arrival}, dv: {dv_dpt}")
                dv_dpt = dv_dpt.to(u.m / u.s)
                dv_dpt = norm(dv_dpt)
                c3_launch = dv_dpt**2
                c3[i, j] = c3_launch.to_value(u.km**2 / u.s**2)
        return c3

    def plotter(self, filename, num_best=10, plot_best=False):
        c3_launch = self._get_c3_launch()
        # print(c3_launch)
        
        if self.ax is None:
            fig, self.ax = plt.subplots(figsize=(15, 15))
        else:
            fig = self.ax.figure
        
        c3_levels = np.linspace(0, self.max_c3.to_value(u.km**2 / u.s**2), 30)
        c = self.ax.contourf(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            c3_launch,
            c3_levels
        )
        line = self.ax.contour(
            [D.to_datetime() for D in self.launch_span],
            [A.to_datetime() for A in self.arrival_span],
            c3_launch,
            c3_levels,
            colors = "black",
            linestyles = "solid"
        )
        cbar = fig.colorbar(c)
        cbar.set_label("km2 / s2")
        self.ax.clabel(line, inline = 1, fmt = "%1.1f", colors = "k", fontsize = 10)
        self.ax.grid()
        fig.autofmt_xdate()
        self.ax.set_title(
            f"Earth - Mars for year {self.launch_span[0].datetime.year}, C3 Launch",
            fontsize = 14,
            fontweight = "bold"
        )
        self.ax.set_xlabel("Launch date", fontsize = 10, fontweight = "bold")
        self.ax.set_ylabel("Arrival date", fontsize = 10, fontweight = "bold")

        c3_launch = c3_launch * u.km**2 / u.s**2

        def ind_to_row_col(ind):
            row = ind // len(self.launch_span)
            col = ind % len(self.launch_span)
            return row, col

        c3_launch_arr = np.array(c3_launch)
        best = [ind_to_row_col(x) for x in np.argsort(c3_launch_arr.flatten())[:num_best]]
        best_launch = [self.launch_dates[p[1]] for p in best]
        best_arrival = [self.arrival_dates[p[0]] for p in best]

        if plot_best:
            if num_best > 1:
                lab = f"{num_best} lowest launch energies"
            else:
                lab = f"Launch date: {best_launch[0]}\nArrival date: {best_arrival[0]}"
            self.ax.scatter(best_launch, best_arrival, color="red", label=lab)
            self.ax.legend(loc="best")
        
        plt.savefig(f"ours_{filename}.svg", bbox_inches="tight")
        return best_launch, best_arrival
