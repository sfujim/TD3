import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline



# Function add timesteps to evaluations
def t_mul(n):
    return n * 3e3

# Load data sources
data = np.load('results/td3/15e3longsuc.npy')
# data = data[:2500]


# Add timesteps
xi = list(range(len(data)))

# splp = make_interp_spline(xi, P, k=1)
# p_smooth = splp(xnew)

plt.plot(xi ,data)

plt.legend()




plt.xlabel('Episodes\n (after 15e3 initial random actions to train VAE)')
plt.ylabel('Reward per Episode')

# TODO: label the data
# TODO: label the axis
plt.show()