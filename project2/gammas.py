import numpy as np
from temperature import temp_to_neutrino_temp
from scipy.integrate import simpson


def gamma_neutron_proton(log_temp, compute_neutron_to_proton=False):
    """
    Computes decay rates for either protons to neutrons or neutrons to protons
    """
    q = -2.53
    if compute_neutron_to_proton:
        q = -q
    FREE_NEUTRON_DECAY_TIME = 1700 #s
    integral_lim__min = 1
    integral_lim_max = 100
    num_elements = 1001
    assert num_elements % 2 != 0, "num points must be odd"
    x = np.linspace(integral_lim__min, integral_lim_max, num_elements)
    temp = np.exp(log_temp)
    neutrino_temp = temp_to_neutrino_temp(temp)
    temp_9 = temp*1e-9
    neutrino_temp_9 = neutrino_temp * 1e-9
    Z = 5.93/temp_9
    Z_nu = 5.93/neutrino_temp_9
    neutron_proton_integrand = (
        (x + q)**2*(x**2 - 1)**(1/2)*x
        / (1 + np.exp(x*Z))/(1 + np.exp(-(x + q)*Z_nu))
        + (x - q)**2*(x**2 - 1)**(1/2)*x
        / (1 + np.exp(-x*Z))/(1 + np.exp((x - q)*Z_nu))
    )
    integral = simpson(neutron_proton_integrand, x)
    assert isinstance(integral, float)
    return integral/FREE_NEUTRON_DECAY_TIME


def gamma_proton_to_neutron(log_temp):  
    """
    Compute decay rate for protons to neutrons
    """
    return gamma_neutron_proton(log_temp)


def gamma_neutron_to_proton(log_temp):
    """
    Compute decay rate for neutrons to protons
    """
    return gamma_neutron_proton(log_temp, compute_neutron_to_proton=True)
