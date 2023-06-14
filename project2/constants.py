import numpy as np
import astropy.constants as const
from astropy import units as u

# g
PROTON_MASS = const.m_p.cgs.value

# g
NEUTRON_MASS = const.m_n.cgs.value

# g
ELECTRON_MASS = const.m_e.cgs.value

#km/s/mpc
H0 = 100*0.7
# km/s/Mpc to 1/s conversion factor
CONVERSION_FACTOR = 1 / (3.08567758e19 * 1e-3)

HUBBLE_PARAM_TODAY = H0 * CONVERSION_FACTOR

# erg/K
BOLTZMANN_CONST = const.k_B.cgs.value

# erg*s
HBAR = const.hbar.cgs.value

# cm/s
C = const.c.cgs.value

# dyn * cm^2/g^2
GRAV_CONST = const.G.cgs.value

# K
PHOTON_TEMP_TODAY = 2.725

# g/cm^3
CRIT_EN_DN_TODAY = (
    3 * HUBBLE_PARAM_TODAY**2 / (8 * np.pi * GRAV_CONST)
)

NUM_NEUTRINO_SPECIES = 3

# erg^4/(cm^5 * s^9)
RAD_EN_DN_TODAY = (
    np.pi ** 2 * BOLTZMANN_CONST ** 4 * PHOTON_TEMP_TODAY ** 4 /
    (15 * HBAR ** 3 * C ** 5) *
    (1 + NUM_NEUTRINO_SPECIES * 7 / 8 * (4 / 11) ** (4 / 3))
)

# erg^4/(g * cm^3 * s^2)
RAD_FRAC_TODAY = RAD_EN_DN_TODAY / CRIT_EN_DN_TODAY

BARYON_MASS_DENSITY_TODAY = 0.05 * CRIT_EN_DN_TODAY

def main():
    print(f"PROTON_MASS: {PROTON_MASS:.4g}")
    print(f"NEUTRON_MASS: {NEUTRON_MASS:.4g}")
    print(f"ELECTRON_MASS: {ELECTRON_MASS:.4g}")
    print(f"HUBBLE_PARAM_TODAY: {HUBBLE_PARAM_TODAY:.4g}")
    print(f"BOLTZMANN_CONST: {BOLTZMANN_CONST:.4g}")
    print(f"HBAR: {HBAR:.4g}")
    print(f"C: {C:.4g}")
    print(f"GRAV_CONST: {GRAV_CONST:.4g}")
    print(f"PHOTON_TEMP_TODAY: {PHOTON_TEMP_TODAY:.4g}")
    print(f"CRIT_EN_DN_TODAY: {CRIT_EN_DN_TODAY:.4g}")
    print(f"NUM_NEUTRINO_SPECIES: {NUM_NEUTRINO_SPECIES:.4g}")
    print(f"RAD_EN_DN_TODAY: {RAD_EN_DN_TODAY:.4g}")
    print(f"RAD_FRAC_TODAY: {RAD_FRAC_TODAY:.4g}")
    print(f"BARYON_MASS_DENSITY_TODAY: {BARYON_MASS_DENSITY_TODAY:.4g}")

if __name__ == "__main__":
    main()