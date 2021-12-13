import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation

softening_param = 0
#km, kg, s
sun = {
    "m": 1.9884754159566474e+30,
    "p": (130.12022433639504, 493.88921553478576, 1.9894885155190423),
    "v": (-9.08252417469984e-05, 2.763470714263038e-05, 9.333023252795563e-08)
}
earth = {
    "m": 6.045825576341311e+24,
    "p": (-52532397.16477036, -142290670.7988091, 6714.9380375893525),
    "v": (27.459556210605758, -10.428715463342813, 0.0004283693350210454)
}

mars = {
    "m": 6.417120534329241e+23,
    "p": (91724696.20692892, -189839018.6923888, -6228099.232650615),
    "v": (22.733051422552098, 12.621328917003236, -0.29323856116219776)
}

init = [sun,earth,mars]

G = 6.67408e-20
num_bodies = len(init)

def x_accel(s,i,j):
    m2 = init[j]['m']
    x1 = s[i]
    y1 = s[i+num_bodies]
    z1 = s[i+num_bodies*2]
    x2 = s[j]
    y2 = s[j+num_bodies]
    z2 = s[j+num_bodies*2]
    r = np.linalg.norm(np.array([x2, y2, z2]) - np.array([x1, y1, z1])) + softening_param**2
    return m2*(x2-x1)/(r)**(3)

def y_accel(s,i,j):
    m2 = init[j]['m']
    x1 = s[i]
    y1 = s[i+num_bodies]
    z1 = s[i+num_bodies*2]
    x2 = s[j]
    y2 = s[j+num_bodies]
    z2 = s[j+num_bodies*2]
    r = np.linalg.norm(np.array([x2, y2, z2]) - np.array([x1, y1, z1])) + softening_param**2
    return m2*(y2-y1)/(r)**(3)


def z_accel(s,i,j):
    m2 = init[j]['m']
    x1 = s[i]
    y1 = s[i+num_bodies]
    z1 = s[i+num_bodies*2]
    x2 = s[j]
    y2 = s[j+num_bodies]
    z2 = s[j+num_bodies*2]
    r = np.linalg.norm(np.array([x2, y2, z2]) - np.array([x1, y1, z1])) + softening_param**2
    return m2*(z2-z1)/(r)**(3)

def xpp(s,i):
    other_indices = list(range(num_bodies))
    other_indices.remove(i)
    return G*sum([x_accel(s,i,j) for j in other_indices])

def ypp(s,i):
    other_indices = list(range(num_bodies))
    other_indices.remove(i)
    return G*sum([y_accel(s,i,j) for j in other_indices])

def zpp(s,i):
    other_indices = list(range(num_bodies))
    other_indices.remove(i)
    return G*sum([z_accel(s,i,j) for j in other_indices])

def F(s,t):
    return np.concatenate(
        (s[3*num_bodies:4*num_bodies],s[4*num_bodies:5*num_bodies],s[5*num_bodies:6*num_bodies],
         [xpp(s,i) for i in range(num_bodies)],
         [ypp(s,i) for i in range(num_bodies)],
         [zpp(s,i) for i in range(num_bodies)]))

def plot_3d():
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')
    while True:
        for k in range(1000):
            plt.cla()
            for i in range(num_bodies):
                ax.plot3D(solution[0:k,i]- solution[0:k,0], solution[0:k,num_bodies+i] - solution[0:k,num_bodies], solution[0:k,num_bodies*2 + i]- solution[0:k,num_bodies*2], 'red', linewidth=0.5)
                ax.scatter3D(solution[k, i] - solution[k, 0], solution[k, num_bodies + i]- solution[k, num_bodies], solution[k, num_bodies*2 + i ] -  solution[k, num_bodies*2], 'ko', alpha=.5)
            ax.set_xlim3d(-3e8, 3e8)
            ax.set_ylim3d(-3e8, 3e8)
            ax.set_zlim3d(-3e8, 3e8)
            fig.canvas.draw()
            plt.pause(.0001)

def plot_2d():
    fig = plt.figure(figsize=(6, 6))
    while True:
        for k in range(1000):
            plt.cla()
            for i in range(num_bodies):
                plt.plot(solution[0:k,i]- solution[0:k,0], solution[0:k,num_bodies+i] - solution[0:k,num_bodies], 'red', linewidth=0.5)
                plt.scatter(solution[k, i] - solution[k, 0], solution[k, num_bodies + i]- solution[k, num_bodies], alpha=.5)
            plt.xlim([-3e8, 3e8])
            plt.ylim([-3e8, 3e8])
            fig.canvas.draw()
            plt.pause(.0001)

def animate_outcome(pos, colors, sizes):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))



    def update(i):
        ax.clear()
        ax.set_facecolor("black")
        ax.grid(False)
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        ax.set_ylim(-246360812.31482115, 6e8)
        ax.set_xlim(-190289032.31830737, 227205650.0355562)
        ax.set_zlim(-7199082.133277591, 4000949.6426398293)


        for p in range(num_bodies):
            ax.plot3D(pos[0:i, p] - pos[0:i, 0], pos[0:i, num_bodies + p] - pos[0:i, num_bodies],
                      pos[0:i, num_bodies * 2 + p] - pos[0:i, num_bodies * 2], color="white", ls="--")
            ax.scatter3D(pos[i, p] - pos[i, 0], pos[i, num_bodies + p] - pos[i, num_bodies],
                         pos[i, num_bodies * 2 + p] - pos[i, num_bodies * 2], color=colors[p], s=sizes[p])
    anim = FuncAnimation(fig, update, frames= 1000)
    return anim

if __name__ == "__main__":
    x0 = [o['p'][0] for o in init]
    y0 = [o['p'][1] for o in init]
    z0 = [o['p'][2] for o in init]
    vx0 = [o['v'][0] for o in init]
    vy0 = [o['v'][1] for o in init]
    vz0 = [o['v'][2] for o in init]
    s0 = np.concatenate((x0,y0,z0,vx0,vy0,vz0))

    t = np.linspace(0,3600*24*365*2,79)
    solution = odeint(F,s0,t)
    anim = animate_outcome(solution, ["yellow", "blue" , "red"], [30,20,15])
    plt.show()
    #anim.save("nbody-odeint.gif", writer="imagemagick", fps=2)
