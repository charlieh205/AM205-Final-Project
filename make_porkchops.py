from poliastro_porkchop import PorkchopPlotter
from poliastro.bodies import Earth, Mars
from poliastro.util import time_range

launch_span = time_range("2003-03-01", end="2004-03-01")
arrival_span = time_range("2003-08-01", end="2005-03-01")

porkchop_plot = PorkchopPlotter(Earth, Mars, launch_span, arrival_span, tfl=False, vhp=False)
_, _, c3_launch, _, _, best_launch, best_arrival = porkchop_plot.porkchop("rovers", 1, True)
print(best_launch)
print(best_arrival)