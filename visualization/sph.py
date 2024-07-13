import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import logging.handlers as handlers
WIDTH=20
HEIGHT=10
num_particles = 10

handlers.RotatingFileHandler('sph.log', maxBytes=1000000, backupCount=10)\
    .setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

particles = np.random.rand(num_particles, 2) * min(WIDTH, HEIGHT)
particles_copy = np.copy(particles)

v0 = np.ones((num_particles, 2))*0.1
v0_copy = np.copy(v0)

scatter = plt.scatter(particles[:, 0], particles[:, 1])
rect = patches.Rectangle((0, 0), WIDTH, HEIGHT, linewidth=1, edgecolor='r', facecolor='none')
plt.gca().add_patch(rect)

plt.xlim(0-1, WIDTH+1)
plt.ylim(0-1, HEIGHT+1)


def update(frame_num):
    global particles, v0
    if frame_num == 0:
        particles = particles_copy.copy()
        v0 = v0_copy.copy()

    # update particle positions
    particles += v0

    # boundary conditions, keep particles in the box
    for i in range(num_particles):
        x, y = particles[i]
        if WIDTH <= x or x <= 0:
            v0[i, 0] *= -1

        if HEIGHT <= y or y <= 0:
            v0[i, 1] *= -1

    # particles = np.clip(particles, 0, 10)  # (new data)

    scatter.set_offsets(particles)
    plt.title("Frame #: " + str(frame_num))
    print("Frame #: ", frame_num, particles_copy[0])
    return scatter,


def run():
    ani = FuncAnimation(plt.gcf(), update, frames=10*20,
                        interval=1000*0.1)

    # plt.scatter(partices[:, 0], partices[:, 1])
    plt.show()

    # ani.save('sine_wave_animation.gif', writer='imagemagick', fps=30)


a = np.array([1.0, 2, 3, 4])
b = a.copy()


def test():
    global a
    print(a is b)
    a = b
    print(a is b)
    a += 1.0
    print(a is b)


if __name__ == '__main__':

    # test()

    run()
