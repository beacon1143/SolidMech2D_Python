import numpy as np
import matplotlib.pyplot as plt

# PHYSICS
lX = 10.0
c = -1.0

# NUMERICS
dt = 0.01
nSteps = 100
nX = 100

# PREPROCESSING
x = np.linspace(-0.5 * lX, 0.5 * lX, nX)

# INITIAL CONDITIONS
p0 = 1.0
p = np.exp(-p0 * x * x)

plt.ion()    # interactive mode
graph = plt.plot(x, p)[0]
plt.pause(0.1)

# ACTION LOOP
for i in range(nSteps):
    dpdt = c * p
    p = p + dpdt * dt
    graph.remove()
    graph = plt.plot(x, p)[0]
    plt.title(str(i+1))
    plt.pause(0.001)
