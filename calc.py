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
nSteps = 400
cfl = 0.5

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
graph = plt.pcolormesh(x, y, p)
plt.colorbar()
plt.gca().set_aspect('equal')
plt.pause(0.1)

# ACTION LOOP
for i in range(nSteps):
    divV = np.diff(vX, 1, 0) / dX + np.diff(vY, 1, 1) / dY
    p = p - divV * K * dt
    tauXX = tauXX + (np.diff(vX, 1, 0) / dX - divV / 3.0) * 2.0 * G * dt
    tauYY = tauYY + (np.diff(vY, 1, 1) / dY - divV / 3.0) * 2.0 * G * dt
    tauXY = tauXY + (np.diff(vX[1:-1, :], 1, 1) / dY + np.diff(vY[:, 1:-1], 1, 0) / dY) * G * dt
    dvXdt = (np.diff(-p[:, 1:-1] + tauXX[:, 1:-1], 1, 0) / dX + np.diff(tauXY, 1, 1) / dY) / rho
    vX[1:-1, 1:-1] = vX[1:-1, 1:-1] + dvXdt * dt
    dvYdt = (np.diff(-p[1:-1, :] + tauYY[1:-1, :], 1, 1) / dY + np.diff(tauXY, 1, 0) / dX ) / rho
    vY[1:-1, 1:-1] = vY[1:-1, 1:-1] + dvYdt * dt
    graph.remove()
    graph = plt.pcolormesh(x, y, p)
    plt.title(str(i+1))
    plt.pause(0.001)
