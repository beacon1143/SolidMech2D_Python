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

fig, graph = plt.subplots()
gr = graph.pcolormesh(x, y, T, shading='auto')
graph.axis('scaled')
graph.set_title('T')
fig.colorbar(gr, location='right')

# ACTION LOOP
def action_loop(i):
    qx[:] = -lam * np.diff(T[:, 1:-1], 1, 0) / dx
    qy[:] = -lam * np.diff(T[1:-1, :], 1, 1) / dy
    dTdt = -(np.diff(qx, 1, 0) / dx + np.diff(qy, 1, 1) / dy) / rhocp
    T[1:-1, 1:-1] = T[1:-1, 1:-1] + dTdt * dt
    fig.suptitle(str(i+1))
    gr.set_array(T)
    gr.set_clim([T.min(), T.max()])
    return gr

anim = ani.FuncAnimation(fig=fig, func=action_loop, frames=nsteps, interval=1, repeat=False)
plt.show()
