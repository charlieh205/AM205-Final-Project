import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def init_axes():
    fig = plt.figure()
    ax = plt.axes()

    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.grid(False)
    return fig, ax


def animate_outcome3d(pos, colors, sizes):

    from mpl_toolkits import mplot3d

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    N = pos.shape[0]

    angles = np.linspace(0, 180, pos.shape[1])

    bound_max = 1.05 * np.max(np.abs(pos))

    def update(i):
        ax.clear()
        ax.view_init(45, -1 * angles[i])
        ax.set_facecolor("black")
        ax.grid(False)
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        ax.set_ylim(-bound_max, bound_max)
        ax.set_xlim(-bound_max, bound_max)
        ax.set_zlim(-bound_max, bound_max)

        x = pos[:, :, 0]
        y = pos[:, :, 1]
        z = pos[:, :, 2]

        for p in range(N):
            ax.plot(x[p, :i], y[p, :i], z[p, :i], color="white", ls="--")
            ax.scatter(x[p, i], y[p, i], z[p, i], color=colors[p], s=sizes[p])

    anim = FuncAnimation(fig, update, frames=range(0, pos.shape[1], 5))
    return anim


def plot_trajectory(ax, pos, color="white", ls="--", **kwargs):
    x = pos[:, 0]
    y = pos[:, 1]
    ax.plot(x, y, color=color, ls=ls, **kwargs)


def plot_trajectories(pos, colors, sizes, alphas=None):

    N = pos.shape[0]

    fig, ax = init_axes()

    bound_max = 1.25 * max(np.max(np.abs(pos[:, :, 0])), np.max(np.abs(pos[:, :, 1])))
    ax.set_aspect("equal")
    ax.set_xlim((-bound_max, bound_max))
    ax.set_ylim((-bound_max, bound_max))

    x = pos[:, :, 0]
    y = pos[:, :, 1]
    for p in range(N):
        ax.scatter(x[p, -1], y[p, -1], color=colors[p], ls="--", zorder=3)
        ax.plot(x[p], y[p], color="white", ls="--")


def animate_outcome2d(pos, colors, sizes, alphas=None, skip=1):

    if alphas is None:
        alphas = [1 for _ in range(len(colors))]

    fig, ax = init_axes()

    bound_max = 1.25 * max(np.max(np.abs(pos[:, :, 0])), np.max(np.abs(pos[:, :, 1])))

    N = pos.shape[0]

    def update(i):
        ax.clear()

        ax.set_facecolor("black")
        ax.grid(False)

        ax.set_xlim((-bound_max, bound_max))
        ax.set_ylim((-bound_max, bound_max))

        x = pos[:, :, 0]
        y = pos[:, :, 1]

        for p in range(N):
            ax.plot(x[p, :i], y[p, :i], color="white", ls="--", alpha=alphas[p])
            ax.scatter(x[p, i], y[p, i], color=colors[p], s=sizes[p], alpha=alphas[p])

    anim = FuncAnimation(fig, update, frames=range(0, pos.shape[1], skip))
    return anim


if __name__ == "__main__":
    pos = np.load("total.npy")[:9]
    colors = ["yellow", "blue", "red"] + ["silver" for i in range(pos.shape[0] - 3)]
    sizes = [100, 30, 30] + [2 for i in range(pos.shape[0] - 3)]
    anim = animate_outcome3d(pos, colors, sizes)
    anim.save("mars.gif", writer="imagemagick", fps=15)
