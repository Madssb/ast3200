import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
""" self defined imports """
from constants import NEUTRON_MASS, PROTON_MASS, C, BOLTZMANN_CONST
from differentials import differentials_everything as differentials
def main():
    """
    Solve for relative number abundances for protons, neutrons, deuterium, trinium,
    helium 3, helium 4, lithium 7, and beryllium 8. Visualizes their evolution
    as functions of time.
    """
    INIT_TEMP = 100e9 # K
    INIT_LOG_TEMP = np.log(100e9)
    FINAL_LOG_TEMP = np.log(0.01e9)
    Y_n0 = (\
        1 
        / (
        1 + np.exp((NEUTRON_MASS - PROTON_MASS)
        * C**2 / BOLTZMANN_CONST / INIT_TEMP)
        )
        )
    Y_p0 = 1 - Y_n0
    init_params = np.zeros(8)
    init_params[0] = Y_n0
    init_params[1] = Y_p0
    instance = solve_ivp(differentials, [INIT_LOG_TEMP, FINAL_LOG_TEMP], init_params, method="Radau", rtol=1e-9, atol=1e-9)
    log_temp_array = instance.t
    assert len(log_temp_array) > 1, "array is just 1, not good."
    temp_array = np.exp(log_temp_array)
    Y_n = instance.y[0]
    Y_p = instance.y[1]
    Y_D = instance.y[2]
    Y_T = instance.y[3]
    Y_He3 = instance.y[4]
    Y_He4 = instance.y[5]
    Y_Li7 = instance.y[6]
    Y_Be7 = instance.y[7]

    fig, ax = plt.subplots()

    ax.plot(temp_array, Y_n, label="n")
    ax.plot(temp_array, Y_p, label="p")
    ax.plot(temp_array, 2*Y_D, label="D")
    ax.plot(temp_array, 3*Y_T, label="T")
    ax.plot(temp_array, 3*Y_He3, label="He3")
    ax.plot(temp_array, 4*Y_He4, label="He4")
    ax.plot(temp_array, 7*Y_Li7, label="Li7")
    ax.plot(temp_array, 7*Y_Be7, label="Be7")

    ax.set_xlabel("Photon temperature [K]")
    ax.set_ylabel("Relative number density")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01e9, 100e9)
    ax.set_ylim(1e-11, 2)
    ax.invert_xaxis()
    ax.legend()

    fig.savefig("i.pdf")
    plt.show()

if __name__ == "__main__":
    main()