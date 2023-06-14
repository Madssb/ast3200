import numpy as np
from scipy.integrate import simps

def integrand(x):
    # define the integrand function here
    return np.sin(x) / x

a = 1.0  # lower limit of integration
b = 10.0  # upper limit of integration
n = 99  # number of intervals (odd number)

x = np.linspace(a, b, n+1)
y = integrand(x)

result = simps(y, x)

print("Numerical result =", result)
