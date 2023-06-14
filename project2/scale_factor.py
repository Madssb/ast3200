import numpy as np
from constants import RAD_FRAC_TODAY, PHOTON_TEMP_TODAY
from temperature import time_to_temp
def time_to_scale_fac(time):
    """
    Compute scale factor as function of time.
    Main use is in func "log_temp_to_scale_fac()".

    Args:
        time (float): Time since big bang.
    
    Returns:
        (float): Scale factor.
    """
    return np.sqrt(2 * np.sqrt(RAD_FRAC_TODAY)*time)



def log_temp_to_scale_fac(log_temp):
    """
    Compute scale factor as function of the logarithm for photon temperature.
    DOESN'T PROVIDE CORRECT RESULTS SOMEHOW.

    Args:
        log_temp (float): natural Logarithm for photon temperature.

    Returns:
        (float): scale factor.
    """
    temp = np.exp(log_temp)
    time = time_to_temp(temp)
    return time_to_scale_fac(time)


def log_temp_to_scale_fac_v2(log_temp):
    """
    Compute scale factor as a functio nof the logarithm for photon temperature.

    Args:
        log_temp (float): natural Logarithm for photon temperature.

    Returns:
        (float): scale factor.
    """
    temp = np.exp(log_temp)
    return PHOTON_TEMP_TODAY/temp