import numpy as np
import matplotlib.pyplot as plt

# PHYSICS
lX = 10.0
lY = 10.0
K = 1.0
G = 0.5
rho = 1.0

# NUMERICS
nX = 200
nY = 200
nSteps = 500
cfl = 0.5
dmp = 4.0 / nX

# PREPROCESSING
x = np.linspace(-0.5 * lX, 0.5 * lX, nX)
y = np.linspace(-0.5 * lY, 0.5 * lY, nY)
x, y = np.meshgrid(x, y, indexing='ij')    # 1D arrays x and y became 2D
dX = lX / (nX - 1)
dY = lY / (nY - 1)
dt = cfl * min(dX, dY) / np.sqrt((K + 4.0 * G / 3.0) / rho)

# INITIAL CONDITIONS
p0 = 1.0
p = np.exp(-p0 * (x * x + y * y))
tauXX = np.zeros((nX, nY))
tauYY = np.zeros((nX, nY))
tauXY = np.zeros((nX - 1, nY - 1))
vX = np.zeros((nX + 1, nY))
vY = np.zeros((nX, nY + 1))

plt.ion()    # interactive mode
fig, graph = plt.subplots(1, 2)
gr0 = graph[0].pcolormesh(x, y, p)

graph[0].set_xlim(-0.5 * lX, 0.5 * lX)
graph[0].set_ylim(-0.5 * lY, 0.5 * lY)
graph[0].axis('scaled')
graph[0].set_title('p')
fig.colorbar(gr0)

gr1 = graph[1].pcolormesh(x, y, tauXX)
fig.colorbar(gr1)
graph[1].set_xlim(-0.5 * lX, 0.5 * lX)
graph[1].set_ylim(-0.5 * lY, 0.5 * lY)
graph[1].axis('scaled')
graph[1].set_title('tau_xx')
plt.pause(0.1)

# ACTION LOOP
for i in range(nSteps):
    divV = np.diff(vX, 1, 0) / dX + np.diff(vY, 1, 1) / dY
    p = p - divV * K * dt
    tauXX = tauXX + (np.diff(vX, 1, 0) / dX - divV / 3.0) * 2.0 * G * dt
    tauYY = tauYY + (np.diff(vY, 1, 1) / dY - divV / 3.0) * 2.0 * G * dt
    tauXY = tauXY + (np.diff(vX[1:-1, :], 1, 1) / dY + np.diff(vY[:, 1:-1], 1, 0) / dY) * G * dt
    dvXdt = (np.diff(-p[:, 1:-1] + tauXX[:, 1:-1], 1, 0) / dX + np.diff(tauXY, 1, 1) / dY) / rho
    vX[1:-1, 1:-1] = (1 - dmp) * vX[1:-1, 1:-1] + dvXdt * dt
    dvYdt = (np.diff(-p[1:-1, :] + tauYY[1:-1, :], 1, 1) / dY + np.diff(tauXY, 1, 0) / dX ) / rho
    vY[1:-1, 1:-1] = (1 - dmp) * vY[1:-1, 1:-1] + dvYdt * dt
    gr0.remove()
    gr1.remove()
    gr0 = graph[0].pcolormesh(x, y, p)
    gr1 = graph[1].pcolormesh(x, y, tauXX)
    fig.suptitle(str(i+1))
    plt.pause(0.0001)
