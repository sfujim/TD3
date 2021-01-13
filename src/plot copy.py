import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline





# Load data sources
data = np.load('results/keep/ER.npy')

data2 = np.load('results/keep/NoER.npy')
diff = len(data2) - len(data)

print(data.shape)

data = np.concatenate((data, [1000 for x in range(diff)]))


# Add timesteps
xi = list(range(len(data)))

# splp = make_interp_spline(xi, P, k=1)
# p_smooth = splp(xnew)

plt.plot(xi ,data, label="Start delay 15e3")
plt.plot(xi ,data2, label="Random action")

plt.legend()




plt.xlabel('Episodes Count')
plt.ylabel('Reward per Episode')

# TODO: label the data
# TODO: label the axis
plt.show()