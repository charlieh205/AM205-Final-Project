
import numpy as np
from rocket  import animate_outcome
import matplotlib.pyplot as plt

pos = np.load("total.npy")[:9]

colors = ["yellow", "blue", "red"] + ["silver" for i in range(pos.shape[0] - 3)]
sizes = [100, 30, 30] + [2 for i in range(pos.shape[0] - 3)]
anim = animate_outcome(pos, colors, sizes)
anim.save('mars.gif', writer='imagemagick', fps=15)

