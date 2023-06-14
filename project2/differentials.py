import numpy as np
from gammas import gamma_neutron_to_proton, gamma_proton_to_neutron
from hubble_parameter import hubble_param
from baryon_mass_density import baryon_mass_density

def differentials_n_p(log_temp, y):
    """
    Compute the differentials for the relative number densities of neutrons and protons
    with respect to the natural logarithm of photon temperature T.

    Args:
        log_temp (float): Photon temperature [K].
        y (array-like): Array containing the relative number densities of
            neutrons and protons.
    
    Returns:
        (array-like): Array containing the differentials for the relative
            number densities of neutrons and protons.
    """
    Y_n, Y_p = y
    assert isinstance(log_temp, float)
    assert all(isinstance(val, float) for val in y)
    assert len(y) == 2
    assert np.abs(np.sum(y) - 1) < 1e-2, f"expected 1, got {np.abs(np.sum(y) - 1)}"
    hubble_param_val = hubble_param(log_temp)
    gamma_proton_to_neutron_val = gamma_proton_to_neutron(log_temp)
    gamma_neutron_to_proton_val = gamma_neutron_to_proton(log_temp)
    dY_n = (
        - Y_n * gamma_neutron_to_proton_val
        + Y_p * gamma_proton_to_neutron_val 
    )
    dY_p = (
        Y_n * gamma_neutron_to_proton_val 
        - Y_p * gamma_proton_to_neutron_val
    )
    return -np.asfarray([dY_n, dY_p])/hubble_param_val

def differentials_n_p_d(log_temp, y):
    """
    Compute the differentials for the relative number densities of neutrons, protons,
    and deuterium with respect to the natural logarithm of photon temperature T.

    Args:
        log_temp (float): Photon temperature [K].
        y (array-like): Array containing the relative number densities of
            neutrons, protons, and deuterium.
    
    Returns:
        (array-like): Array containing the differentials for the relative
            number densities of neutrons, protons and deuterium.
    """
    Y_n, Y_p, Y_D = y
    print(f"{y[0]=:.4g}")
    assert isinstance(log_temp, float)
    assert all(isinstance(val, float) for val in y)
    assert len(y) == 3
    sum = Y_n + Y_p + 2*Y_D
    assert np.abs(sum - 1) < 1e-2, f"expected 1, got {np.abs(np.sum(y) - 1)}"
    temp = np.exp(log_temp)
    temp9 = temp*1e-9
    baryon_mass_density_val = baryon_mass_density(log_temp)
    hubble_param_val = hubble_param(log_temp)
    lambda_wp = gamma_proton_to_neutron(log_temp)
    lambda_wn = gamma_neutron_to_proton(log_temp)
    dY_n = - Y_n * lambda_wn + Y_p * lambda_wp
    dY_p = Y_n * lambda_wn - Y_p * lambda_wp
    #(1)    p + n <--> D + gamma
    pn = baryon_mass_density_val*2.5e4
    lambda_D = 4.68e9*pn/baryon_mass_density_val*temp9**(3/2)*np.exp(-25.82/temp9)
    dY_n += lambda_D*Y_D - pn*Y_n*Y_p
    dY_p += lambda_D*Y_D - pn*Y_n*Y_p
    dY_D = -lambda_D*Y_D + pn*Y_n*Y_p
    differentials = np.asarray([dY_n, dY_p, dY_D])  
    return -differentials/hubble_param_val



def differentials_everything(log_temp, y):
    """
    Compute the differentials for the relative number densities of neutrons, protons,
    deuterium, trinium, helium 3, helium 4, lithium 7, beryllium 7, with respect
    to the natural logarithm of photon temperature T.

    This model is ultimately unsuccesful, and results can't be trusted.

    Args:
        log_temp (float): Photon temperature [K].
        y (array-like): Array containing the relative number densities of
            neutrons, protons, deuterium, trinium, helium 3, helium 4,
            lithium 7, and beryllium 7.
    
    Returns:
        (array-like): Array containing the differentials for the relative
            number densities of neutrons, protons and deuterium.
    """
    Y_n = y[0]
    Y_p = y[1]
    Y_D = y[2]
    Y_T = y[3]
    Y_He3 = y[4]
    Y_He4 = y[5]
    Y_Li7 = y[6]
    Y_Be7 = y[7]
    dY_n = 0
    dY_p = 0
    dY_D = 0
    dY_T = 0
    dY_He3 = 0
    dY_He4 = 0
    dY_Li7 = 0
    dY_Be7 = 0
    assert isinstance(log_temp, float)
    assert all(isinstance(val, float) for val in y)
    assert len(y) == 8
    atomic_mass_numbers = [1,1,2,3,3,4,7,7] 
    assert np.abs(np.sum(y*atomic_mass_numbers) - 1) < 1e-2, f"{np.abs(np.sum(y*atomic_mass_numbers) - 1)}"
    temp = np.exp(log_temp)
    temp9 = temp*1e-9 
    # weak interactions
    #(1)    n + nu_e <--> p + e-
    #(2)    n + e+ <--> p + bar(nu_e) (??) 
    #(3)    n <--> p + e- +bar(nu_e) (??)
    lambda_wp = gamma_proton_to_neutron(log_temp)
    lambda_wn = gamma_neutron_to_proton(log_temp)
    dY_n += Y_p * lambda_wp - Y_n * lambda_wn
    dY_p -= Y_p * lambda_wp - Y_n * lambda_wn
    baryon_mass_density_val = baryon_mass_density(log_temp)
    hubble_param_val = hubble_param(log_temp)
    #strong and EM interactions
    #(1)    p + n <--> D + gamma
    pn = baryon_mass_density_val*2.5e4  
    lambda_D = 4.68e9*pn/baryon_mass_density_val*temp9**(3/2)*np.exp(-25.82/temp9)
    dY_n += lambda_D*Y_D - pn*Y_n*Y_p
    dY_p += lambda_D*Y_D - pn*Y_n*Y_p
    dY_D -= lambda_D*Y_D - pn*Y_n*Y_p    
    #(2)    p + D <--> He3 + gamma
    pD = 2.23e3*baryon_mass_density_val*temp9**(-2/3)*np.exp(-3.72*temp9**(-1/3))*(1 + 0.112*temp9**(1/3) + 3.38*temp9**(2/3) + 2.65*temp9)
    lambda_He3 = 1.63e10*pD/baryon_mass_density_val*temp9**(3/2)*np.exp(-63.75/temp9)
    dY_p += lambda_He3*Y_He3 - pD*Y_p*Y_D
    dY_D += lambda_He3*Y_He3 - pD*Y_p*Y_D
    dY_He3 -= lambda_He3*Y_He3 - pD*Y_p*Y_D
    #(3)    n + D <--> T + gamma
    nD = baryon_mass_density_val*(75.5 + 1250*temp9)
    lambda_T = 1.63e10*nD/baryon_mass_density_val*temp9**(3/2)*np.exp(-72.62/temp9)
    dY_n += lambda_T*Y_T - nD*Y_n*Y_D
    dY_D += lambda_T*Y_T - nD*Y_n*Y_D
    dY_T -= lambda_T*Y_T - nD*Y_n*Y_D
    #(4)    n + He3 <--> p + T
    nHe3_p = 7.06e8*baryon_mass_density_val
    pT_n = nHe3_p*np.exp(-8.864/temp9)
    dY_n += pT_n*Y_p*Y_T - nHe3_p*Y_n*Y_He3
    dY_He3 += pT_n*Y_p*Y_T - nHe3_p*Y_n*Y_He3
    dY_p -= pT_n*Y_p*Y_T - nHe3_p*Y_n*Y_He3
    dY_T -= pT_n*Y_p*Y_T - nHe3_p*Y_n*Y_He3
    #(5)    p + T <--> He4 + gamma
    pT_gamma = 2.87e4*baryon_mass_density_val*temp9**(-2/3)*np.exp(-3.87*temp9**(1/3))*(1 + 0.108*temp9**(1/3) + 0.466*temp9**(2/3) + 0.352*temp9 + 0.300*temp9**(4/3) + 0.576*temp9**(5/3))
    lambda_He4_p = 2.59e10*pT_gamma/baryon_mass_density_val*temp9**(3/2)*np.exp(-229.9/temp9)
    dY_p += lambda_He4_p*Y_He4 - pT_gamma*Y_p*Y_T
    dY_T += lambda_He4_p*Y_He4 - pT_gamma*Y_p*Y_T
    dY_He4 -= lambda_He4_p*Y_He4 - pT_gamma*Y_p*Y_T
    #(6)    n + He3 <--> He4 + gamma
    nHe3_gamma = 6.0e3*baryon_mass_density_val*temp9
    lambda_He4_n = 2.60e10*nHe3_gamma/baryon_mass_density_val*temp9**(3/2)*np.exp(-238.8/temp9)
    dY_n += lambda_He4_n*Y_He4 - nHe3_gamma*Y_n*Y_He3
    dY_He3 += lambda_He4_n*Y_He4 - nHe3_gamma*Y_n*Y_He3
    dY_He4 -= lambda_He4_n*Y_He4 - nHe3_gamma*Y_n*Y_He3
    #(7)    D + D <--> n + He3
    DD_n = 3.9e8*baryon_mass_density_val*temp9**(-2/3)*np.exp(-4.26*temp9**(-1/3))*(1 + 0.0979*temp9**(1/3) + 0.642*temp9**(2/3) + 0.440*temp9)
    nHe3_D = 1.73*DD_n*np.exp(-37.94/temp9)
    dY_D += 2*nHe3_D*Y_n*Y_He3 - DD_n*Y_D*Y_D
    dY_n -= nHe3_D*Y_n*Y_He3 -DD_n*Y_D*Y_D/2
    dY_He3 -= nHe3_D*Y_n*Y_He3 -DD_n*Y_D*Y_D/2
    #(8) D + D <--> p + T
    DD_p = DD_n
    pT_D = 1.73*DD_p*np.exp(-46.80/temp9)
    dY_D += 2*pT_D*Y_T*Y_p - DD_p*Y_D*Y_D
    dY_p -= pT_D*Y_T*Y_p - DD_p*Y_D*Y_D/2
    dY_T -= pT_D*Y_T*Y_p - DD_p*Y_D*Y_D/2
    #(9) D + D <--> He4 + gamma
    DD_gamma = 24.1*baryon_mass_density_val*temp9**(-2/3)*np.exp(-4.26*temp9**(-1/3))*(temp9**(2/3) + 0.685*temp9 + 0.152*temp9**(4/3) + 0.265*temp9**(5/3))
    lambda_He4_D = 4.50e10*DD_gamma/baryon_mass_density_val*temp9**(3/2)*np.exp(-276.7/temp9)
    dY_D += 2*lambda_He4_D*Y_He4 - DD_gamma*Y_D*Y_D
    dY_He4 -= lambda_He4_D*Y_He4 - DD_gamma*Y_D*Y_D/2
    #(10) D + He3 <--> He4 + p
    DHe3 = 2.60e9*baryon_mass_density_val*temp9**(-3/2)*np.exp(-2.99/temp9)
    He4p = 5.50*DHe3*np.exp(-213.0/temp9)

    dY_D += He4p*Y_He4*Y_p - DHe3*Y_D*Y_He3
    dY_He3 += He4p*Y_He4*Y_p - DHe3*Y_D*Y_He3
    dY_He4 -= He4p*Y_He4*Y_p - DHe3*Y_D*Y_He3
    dY_p -= He4p*Y_He4*Y_p - DHe3*Y_D*Y_He3
    #(11) D + T <--> He4 + n
    DT = 1.38e9*baryon_mass_density_val*temp9**(-3/2)*np.exp(-0.745/temp9)
    He4n = 5.50*DT*np.exp(-204.1/temp9)
    dY_D += He4n*Y_He4*Y_n - DT*Y_D*Y_T
    dY_T += He4n*Y_He4*Y_n - DT*Y_D*Y_T
    dY_He4 -= He4n*Y_He4*Y_n - DT*Y_D*Y_T
    dY_n -= He4n*Y_He4*Y_n - DT*Y_D*Y_T
    #(15)   He3 + T <--> He4 + D
    He3T_D = 3.88e9 *baryon_mass_density_val*temp9**(-2/3)*np.exp(-7.72*temp9**(-1/3))*(1 + 0.0540*temp9**(1/3))
    He4_D = 1.59*He3T_D*np.exp(-166.2/temp9)
    dY_He3 += He4_D*Y_He4*Y_D - He3T_D*Y_He3*Y_T
    dY_T += He4_D*Y_He4*Y_D - He3T_D*Y_He3*Y_T
    dY_He4 -= He4_D*Y_He4*Y_D - He3T_D*Y_He3*Y_T
    dY_D -= He4_D*Y_He4*Y_D - He3T_D*Y_He3*Y_T
    #(16)   He3 + He4 <--> Be7 + gamma
    He3He4 = 4.8e6 * baryon_mass_density_val * temp9**(-2/3) * np.exp(-12.8 * temp9**(-1/3)) * (1 + 0.0326 * temp9**(1/3) - 0.219 * temp9**(2/3) - 0.0499 * temp9 + 0.0258 * temp9**(4/3) + 0.0150 * temp9**(5/3))
    lambda_Be7 = 1.12e10*He3He4/baryon_mass_density_val*temp9**(3/2)*np.exp(-18.42/temp9)
    dY_He3 += lambda_Be7*Y_Be7 - He3He4*Y_He3*Y_He4
    dY_He4 += lambda_Be7*Y_Be7 - He3He4*Y_He3*Y_He4
    dY_Be7 -= lambda_Be7*Y_Be7 - He3He4*Y_He3*Y_He4
    #(17)   T + He4 <--> Li7 + gamma
    THe4 = 5.28e5 * baryon_mass_density_val * temp9**(-2/3) * np.exp(-8.08 * temp9**(-1/3)) * (1 + 0.0516 * temp9**(1/3))
    lambda_Li7 = 1.12e10*THe4/baryon_mass_density_val*temp9**(3/2)*np.exp(-28.63/temp9)
    dY_T += lambda_Li7*Y_Li7 - THe4*Y_T*Y_He4
    dY_He4 += lambda_Li7*Y_Li7 - THe4*Y_T*Y_He4
    dY_Li7 -= lambda_Li7*Y_Li7 - THe4*Y_T*Y_He4
    #(18)   n + Be7 <--> p + Li7
    nBe7_p = 6.74e9*baryon_mass_density_val
    pLi7_n = nBe7_p*np.exp(-19.07/temp9)
    dY_n += pLi7_n*Y_p*Y_Li7 - nBe7_p*Y_n*Y_Be7
    dY_Be7 += pLi7_n*Y_p*Y_Li7 - nBe7_p*Y_n*Y_Be7
    dY_p -= pLi7_n*Y_p*Y_Li7 - nBe7_p*Y_n*Y_Be7
    dY_Li7 -= pLi7_n*Y_p*Y_Li7 - nBe7_p*Y_n*Y_Be7
    #(20)     p + Li7 <--> He4 + He4
    pLi7_He4 = 1.42e9 * baryon_mass_density_val * temp9**(-2/3) * np.exp(-8.47 * temp9**(-1/3)) * (1 + 0.0493 * temp9**(1/3))
    He4He4_p = 4.64 * pLi7_He4 * np.exp(-201.3/temp9)
    dY_p += He4He4_p*Y_He4*Y_He4/2 - pLi7_He4*Y_p*Y_Li7
    dY_Li7 += He4He4_p*Y_He4*Y_He4/2 - pLi7_He4*Y_p*Y_Li7
    dY_He4 -= He4He4_p*Y_He4*Y_He4 - pLi7_He4*Y_p*Y_Li7*2


    #(21)   n + Be7 <--> He4 + He4
    nBe7_He4 = 1.2e7*baryon_mass_density_val*temp9
    He4He4_n = 4.64*nBe7_He4*np.exp(-220.4/temp9)
    dY_n += He4He4_n*Y_He4*Y_He4/2 - nBe7_He4*Y_n*Y_Be7
    dY_Be7 += He4He4_n*Y_He4*Y_He4/2 - nBe7_He4*Y_n*Y_Be7
    dY_He4 -= He4He4_n*Y_He4*Y_He4 - 2*nBe7_He4*Y_n*Y_Be7
    differentials = np.asarray([dY_n, dY_p, dY_D, dY_T, dY_He3, dY_He4, dY_Li7, dY_Be7])
    return -differentials/hubble_param_val
