# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


data = np.load("latent.npy")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = []
ys = []
zs = []

for i in data:
    xs.append(i[0])
    ys.append(i[2])
    zs.append(i[1])
ax.scatter(xs, ys, zs, marker='o')


plt.show()