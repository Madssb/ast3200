import numpy as np
from constants import HUBBLE_PARAM_TODAY, RAD_FRAC_TODAY, PHOTON_TEMP_TODAY
from scale_factor import log_temp_to_scale_fac_v2 as log_temp_to_scale_fac

def hubble_param(log_temp):
    """
    Compute Hubble parameter.
    """
    return HUBBLE_PARAM_TODAY * np.sqrt(RAD_FRAC_TODAY) / \
        log_temp_to_scale_fac(log_temp)**2



if __name__ == "__main__":
    """
    checks if the hubble parameter is computed correctly.
    """
    assert np.abs(hubble_param(PHOTON_TEMP_TODAY) - HUBBLE_PARAM_TODAY) < 1e-3, f"{np.abs(hubble_param(PHOTON_TEMP_TODAY) - HUBBLE_PARAM_TODAY)=:.4g}"