import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits import mplot3d

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
n = len(init)

def x_accel(s,i,j):
    m2 = init[j]['m']
    x1 = s[i]
    y1 = s[i+n]
    z1 = s[i+n+1]
    x2 = s[j]
    y2 = s[j+n]
    z2 = s[j+n+1]
    return m2*(x2-x1)/((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(3/2)

def y_accel(s,i,j):
    m2 = init[j]['m']
    x1 = s[i]
    y1 = s[i+n]
    z1 = s[i+n+1]
    x2 = s[j]
    y2 = s[j+n]
    z2 = s[j+n+1]
    return m2*(y2-y1)/((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(3/2)


def z_accel(s,i,j):
    m2 = init[j]['m']
    x1 = s[i]
    y1 = s[i+n]
    z1 = s[i+n+1]
    x2 = s[j]
    y2 = s[j+n]
    z2 = s[j+n+1]
    return m2*(z2-z1)/((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(3/2)

def xpp(s,i):
    other_indices = list(range(n))
    other_indices.remove(i)
    return G*sum([x_accel(s,i,j) for j in other_indices])

def ypp(s,i):
    other_indices = list(range(n))
    other_indices.remove(i)
    return G*sum([y_accel(s,i,j) for j in other_indices])

def zpp(s,i):
    other_indices = list(range(n))
    other_indices.remove(i)
    return G*sum([z_accel(s,i,j) for j in other_indices])

def F(s,t):
    return np.concatenate(
        (s[2*n:3*n],s[3*n:4*n],s[4*n:5*n],
         [xpp(s,i) for i in range(n)],
         [ypp(s,i) for i in range(n)],
         [zpp(s,i) for i in range(n)]))

x0 = [o['p'][0] for o in init]
y0 = [o['p'][1] for o in init]
z0 = [o['p'][2] for o in init]
vx0 = [o['v'][0] for o in init]
vy0 = [o['v'][1] for o in init]
vz0 = [o['v'][2] for o in init]
s0 = np.concatenate((x0,y0,z0,vx0,vy0,vz0))
print(s0)
print("____________")

t = np.linspace(0,3600*24*365,1000)
solution = odeint(F,s0,t)

def pic(k=0):
    for i in range(n):
        plt.plot(solution[:,i], solution[:,n+i], 'gray', linewidth=0.5)
    for i in range(n):
        plt.plot(solution[k,i], solution[k,n+i], 'ko')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.axis('off');

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
while True:
    for k in range(1000):
        plt.cla()
        for i in range(n):
            ax.plot3D(solution[:, i], solution[:, n + i], np.zeros(len(solution[:, i])), 'red', linewidth=0.5)
        for i in range(n):
            ax.scatter3D(solution[k, i], solution[k, n + i], np.zeros(len(solution[:, i])), 'ko', alpha=.5)
        fig.canvas.draw()
        plt.pause(.01)

