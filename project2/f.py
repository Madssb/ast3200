import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
""" self defined imports """
from constants import NEUTRON_MASS, PROTON_MASS, C, BOLTZMANN_CONST
from differentials import differentials_n_p as differentials

def main():
    """
    Solve for relative number abundances for protons, neutrons, and deuterium.
    Visualizes their evolution as functions of time.
    """
    INIT_TEMP = 100e9 # K
    INIT_LOG_TEMP = np.log(100e9)
    FINAL_LOG_TEMP = np.log(0.1e9)
    INIT_REL_NUM_DN_NEUTRON = (\
        1 
        / (
        1 + np.exp((NEUTRON_MASS - PROTON_MASS)
        * C**2 / BOLTZMANN_CONST / INIT_TEMP)
        )
        )
    INIT_REL_NUM_DN_PROTON = 1 - INIT_REL_NUM_DN_NEUTRON
    INIT_PARAMS  = np.asarray([INIT_REL_NUM_DN_NEUTRON, INIT_REL_NUM_DN_PROTON])
    instance = solve_ivp(differentials, [INIT_LOG_TEMP, FINAL_LOG_TEMP], INIT_PARAMS, method="Radau", rtol=1e-12, atol=1e-12)
    log_temp_array = instance.t
    assert len(log_temp_array) > 1, "array is just 1, not good."
    temp_array = np.exp(log_temp_array)
    rel_num_dns_neut = instance.y[0]
    rel_num_dns_prot = instance.y[1]
    fig, ax = plt.subplots()

    ax.plot(temp_array, rel_num_dns_prot, label="protons")
    ax.plot(temp_array, rel_num_dns_neut, label="neutrons")
    ax.set_xlabel("Photon temperature [K]")
    ax.set_ylabel("Relative number density")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlim(0.1e9,100e9)
    ax.set_ylim(1e-5,2)
    ax.legend()
    fig.savefig("f.pdf")


if __name__ == "__main__":
    main()