"""
Define time_to_temp

"""
import numpy as np
import constants as const


def time_to_temp(temp):
    """
    Compute time since big bang.

    Args:
        temp (float): Photon temperature [K].

    Returns:
        (float): Time.
    """
    return const.PHOTON_TEMP_TODAY**2 / 2 / np.sqrt(const.RAD_FRAC_TODAY) / \
           const.HUBBLE_PARAM_TODAY / temp**2

def temp_to_neutrino_temp(temp):
    """
    Compute the Neutrino temperature [K].
    Args:
        temp (float): Photon temperature [K].
    
    Returns:
        (float): Neutrino temperature [K].
    """
    return (4/11)**(1/3)*temp

def main():
    """
    solves task d.
    """
    print(f"""
universe was {time_to_temp(1e10):.3g}s old at T = {1e10:.3g}K,
{time_to_temp(1e9)/60:.3g}mins old at T = {1e9:.3g}K,
and {time_to_temp(1e8)/3600:.3g}hrs old at T = {1e8:.3g}K.
    """)

if __name__ == "__main__":
    main()