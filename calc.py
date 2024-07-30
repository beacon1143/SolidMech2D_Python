import numpy as np
import matplotlib.pyplot as plt

# PHYSICS
c = -1.0

# NUMERICS
dt = 0.01
nSteps = 100

# PREPROCESSING
t = np.linspace(0, nSteps * dt, nSteps + 1)
p = np.zeros(nSteps + 1)

# INITIAL CONDITIONS
p[0] = 1.0    # p0

plt.ion()    # interactive mode
plt.xlim(0, nSteps * dt)
plt.ylim(0, p[0])
plt.title('0')
graph = plt.plot(t[:1], p[:1])[0]
plt.pause(0.01)

# ACTION LOOP
for i in range(nSteps):
    dpdt = c * p[i]
    p[i + 1] = p[i] + dpdt * dt
    graph.remove()
    graph = plt.plot(t[:i+2], p[:i+2])[0]
    plt.title(str(i+1))
    plt.pause(0.001)
