from constants import BARYON_MASS_DENSITY_TODAY
from scale_factor import log_temp_to_scale_fac_v2


def baryon_mass_density(log_temp):
    """
    Compute baryon mass density.

    Args:
        log_temp (float): Photon temperature [K].

    Returns:
        (float): Baryon mass density [kg/m^3].
    """
    return BARYON_MASS_DENSITY_TODAY*log_temp_to_scale_fac_v2(log_temp)**(-3)