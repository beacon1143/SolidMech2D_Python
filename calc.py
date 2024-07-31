import numpy as np
import matplotlib.pyplot as plt

# PHYSICS
lX = 10.0
K = 1.0
G = 1.5
rho = 1.0

# NUMERICS
nX = 100
nSteps = 1000
cfl = 1.0

# PREPROCESSING
x = np.linspace(-0.5 * lX, 0.5 * lX, nX)
dX = lX / (nX - 1)
dt = cfl * dX / np.sqrt((K + 4.0 * G / 3.0) / rho)

# INITIAL CONDITIONS
p0 = 1.0
p = np.exp(-p0 * x * x)
tau = np.zeros(nX)
v = np.zeros(nX + 1)

plt.ion()    # interactive mode
graph = plt.plot(x, p)[0]
plt.pause(0.1)

# ACTION LOOP
for i in range(nSteps):
    dpdt = -np.diff(v) / dX * K
    p = p + dpdt * dt
    dtaudt = np.diff(v) * G * 4.0 / 3.0
    tau = tau + dtaudt * dt
    dvdt = np.diff(-p + tau) / dX / rho
    v[1:-1] = v[1:-1] + dvdt * dt
    graph.remove()
    graph = plt.plot(x, p)[0]
    plt.title(str(i+1))
    plt.pause(0.001)
