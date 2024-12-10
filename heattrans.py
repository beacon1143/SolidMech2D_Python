import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# PHYSICS
lx = 10.0
ly = 10.0
lam = 1.0
rhocp = 1.0

# NUMERIC
nx = 100
ny = 100
nsteps = 1000
cfl = 1.0

# PREPROCESSING
x = np.linspace(-0.5 * lx, 0.5 * lx, nx)
y = np.linspace(-0.5 * ly, 0.5 * ly, ny)
x, y = np.meshgrid(x, y, indexing='ij')
dx = lx / (nx - 1)
dy = ly / (ny - 1)
dt = cfl * min(dx, dy) ** 2 * rhocp / lam / 4.0

# INITIAL CONDITIONS
T0 = 1.0
T = T0 * np.exp(-x * x - y * y)
qx = np.zeros((nx - 1, nx - 2))
qy = np.zeros((nx - 2, nx - 1))

fig, graph = plt.subplots(1, 3)

def make_plot(fig, plot, x, y, field, name):
    gr = plot.pcolormesh(x, y, field, shading='auto')
    plot.axis('scaled')
    plot.set_title(name)
    fig.colorbar(gr, location='bottom')
    return gr

gr = []
gr.append(make_plot(fig, graph[0], x, y, T, 'T'))
gr.append(make_plot(fig, graph[1], x[:-1, :-2], y[:-1, :-2], qx, 'qx'))
gr.append(make_plot(fig, graph[2], x[:-2, :-1], y[:-2, :-1], qy, 'qy'))

# ACTION LOOP
def action_loop(i):
    qx[:] = -lam * np.diff(T[:, 1:-1], 1, 0) / dx
    qy[:] = -lam * np.diff(T[1:-1, :], 1, 1) / dy
    dTdt = -(np.diff(qx, 1, 0) / dx + np.diff(qy, 1, 1) / dy) / rhocp
    T[1:-1, 1:-1] = T[1:-1, 1:-1] + dTdt * dt
    fig.suptitle(str(i+1))
    gr[0].set_array(T)
    gr[0].set_clim([T.min(), T.max()])
    gr[1].set_array(qx)
    gr[1].set_clim([qx.min(), qx.max()])
    gr[2].set_array(qy)
    gr[2].set_clim([qy.min(), qy.max()])
    return gr

anim = ani.FuncAnimation(fig, action_loop, nsteps, interval=1, repeat=False)
plt.show()
