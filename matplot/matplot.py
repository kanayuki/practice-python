import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

x = np.linspace(0, 2 * np.pi)
y = np.sin(x)

l = plt.plot(x, y)

print(l)
plt.show()

