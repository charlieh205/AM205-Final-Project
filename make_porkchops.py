"""
Porkchop Plot creation CLI tool
"""
from poliastro_porkchop import PorkchopPlotter
from our_porkchop import Porkchop
from poliastro.bodies import Earth, Mars
from poliastro.util import time_range
from datetime import datetime
import argparse


def poliastro(launch_span, arrival_span):
    porkchop_plot = PorkchopPlotter(
        Earth, Mars, launch_span, arrival_span, tfl=False, vhp=False
    )
    _, _, c3_launch, _, _, best_launch, best_arrival = porkchop_plot.porkchop(
        "rovers", 1, True
    )
    print(best_launch)
    print(best_arrival)


def ours(launch_span, arrival_span):
    pork = Porkchop(launch_span, arrival_span)
    best_launch, best_arrival = pork.plotter("rovers", 1, True)
    print(best_launch)
    print(best_arrival)


if __name__ == "__main__":
    choices = {"poliastro": poliastro, "ours": ours}
    parser = argparse.ArgumentParser(
        description="Use either Poliastro or our implementation to make a porkchop plot"
    )
    parser.add_argument(
        "-c",
        metavar="CHOCIE",
        choices=choices,
        default="ours",
        help="Which implementation to use",
    )
    args = parser.parse_args()
    f = choices[args.c]
    launch = time_range("2003-03-01", end="2004-03-01")
    arrival = time_range("2003-08-01", end="2005-03-01")
    f(launch, arrival)
