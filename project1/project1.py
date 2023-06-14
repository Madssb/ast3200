"""
Integrates the equations of motion for x1, x2, x3, and lambda, solving for 
x1, x2 and x3. We do this for the inverse power law, and the exponential respectively.

Plots the density parameters for matter, radiation, and the quintessence as
functions of z.

Also plots the Equation of State w_phi.
"""
import numpy as np
import scipy.constants as c
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def redshift_to_n(z):
    """
    Converts redshift z to N.
    Args:
        z (float or array-like of dtype float): redshift.
    Returns:
        float or array-like of dtype float: N corresponding to redshift.
    """
    n = np.log(1/(z + 1))
    return n


def n_to_redshift(n):
    """
    Converts from time quantity N to redshift
    Args:
        n (float or array-like of dtype float): time quantity N.
    Returns:
        float or array-like of dtype float: redshift corresponding to N.
    """
    return np.exp(-n) - 1


class ComputeQuantities:
    HUBBLE_CONSTANT = 69.8  # (km/s)/Mpc

    def __init__(self, exponential_pot, init_x1, init_x2, init_x3, init_lambda):
        """
        use solve IVP and initial conditions to solve for x_1, x_2, and x_3,
        and derived quantities, i.e. the density parameters, and the EoS
        parameter.
        NB: all arrays operate in correspondence with z = [2e7,...,0]
        
        Args:
            exponential_pot (bool): True when expoential potential is used,
                                    false when power law potential is used.
            init_x1 (float): initial value for x_1.
            init_x2 (float): initial value for x_2.
            init_x3 (float): initial value for x_3.
            init_lambda (float): initial value for lambda.
        Returns:
            None
        """
        assert type(
            exponential_pot) == bool, "Error: exponential_pot is not a boolean"
        self.exponential_pot = exponential_pot
        self.init_vals = np.array([init_x1, init_x2, init_x3, init_lambda])
        solution = self.solve_eqs()
        self.n_array = np.linspace(solution.t[0], solution.t[-1], 10000)
        self.n_sequential = np.flip(self.n_array)
        sol_interpol = solution.sol(self.n_array)
        self.redshift = n_to_redshift(self.n_array)
        self.x_1, self.x_2, self.x_3 = sol_interpol[0], sol_interpol[1], sol_interpol[2]
        self.omega_phi = self.x_1**2 + self.x_2**2
        self.omega_r = self.x_3**2
        self.omega_m = 1 - self.omega_phi - self.omega_r
        # N/m
        self.eos = (self.x_1**2 - self.x_2**2) / (self.x_1**2 + self.x_2**2)
        self.hubble_parameter = self.compute_hubble_parameter()
        self.hubble_sequential = np.flip(self.hubble_parameter)
        self.hubble_parameter_lcdm = self.compute_hubble_lcdm()
        self.hubble_lcdm_sequential = np.flip(self.hubble_parameter_lcdm)
        self.dimless_age = self.compute_dimless_age()
        self.dimless_age_lcdm = self.compute_dimless_age(lcdm=True)
        self.dimless_lum_dist = self.compute_dimless_lum_dist()
        self.dimless_lum_dist_lcdm = self.compute_dimless_lum_dist(lcdm=True)
        self.data = np.loadtxt("data.txt")
        self.chi_squared_ = self.chi_squared()
    

    def diffs(self, N, params):
        """
        Expresses the equation of motions for the dimensionless variables x_1
        x_2, and x_3. the differential for lambda with respect to N is also
        expressed, but only used if the potential used is the power law
        potential, otherwise, overriden by manually defining lambda.

        Args:
            params (ndarray): A NumPy array containing the current
                values of x_1, x_2, x_3, and lambda.

        Returns:
            ndarray: An array of the differentials due to their coupled nature.       
        """
        x_1, x_2, x_3, lambda_ = params[0], params[1], params[2], params[3]
        if self.exponential_pot:
            lambda_ = 3/2
        diff_x_1 = (
            -3 * x_1 + np.sqrt(6) / 2 * lambda_ * x_2**2
            + (1 / 2) * x_1 * (3 + 3 * x_1**2 - 3 * x_2**2 + x_3**2))
        diff_x_2 = (
            (-np.sqrt(6) / 2) * lambda_ * x_1 * x_2
            + (1 / 2) * x_2 * (3 + 3 * x_1**2 - 3 * x_2**2 + x_3**2)
        )
        diff_x_3 = (
            -2 * x_3 + (1 / 2) * x_3 * (3 + 3 * x_1**2 - 3 * x_2**2 + x_3**2)
        )
        diff_lambda = (
            -np.sqrt(6)*lambda_**2*(2-1)*x_1
        )
        diffs = np.array([diff_x_1, diff_x_2, diff_x_3, diff_lambda])
        return diffs

    def solve_eqs(self):
        """
        Solve for x_1, x_2, x_3 and lambda initial value problem by numerical
        integration. solves for z0 =2e7 and towards z=0.
        Returns: 
            Solution object for numerical integration

        """
        time_span_n = redshift_to_n(np.array([2e7, 0]))
        solution = solve_ivp(self.diffs, time_span_n,
                             self.init_vals, rtol=1e-9, atol=1e-9,
                             dense_output=True, method="RK45")
        return solution

    def compute_hubble_parameter(self):
        """
        Compute the (normalized) hubble parameter for the quintessence models.
        NB: it is assumed arrays correspond to a redshift [2e7,0].
        Returns:
            (array-like of dtype float) hubble parameters
        """
        n_low_to_high = np.flip(self.n_array)
        eos_low_to_high = np.flip(self.eos)
        integrand = 3 * (1 + eos_low_to_high)
        integral_flipped = cumulative_trapezoid(
            integrand, n_low_to_high, initial=0)
        integral = np.flip(integral_flipped)
        """
        print(f"{integral[0]=}")
        print(f"{self.omega_m[0]=}")
        print(f"{self.omega_r[0]=}")
        print(f"{self.omega_phi[0]=}")
        print(f"{self.n_array[0]=}")
        """
        hubble_parameter = np.sqrt(
            np.exp(-3*self.n_array)*self.omega_m[-1]
            + np.exp(-4*self.n_array)*self.omega_r[-1]
            + np.exp(integral)*self.omega_phi[-1]
        )
        #print(f"{hubble_parameter[0]=}")
        return hubble_parameter

    def compute_hubble_lcdm(self):
        """
        Compute the hubble parameter for the LCDM model
        Returns:
            (float) Hubble parameter for the LCDM model
        """
        omega_m0 = 0.3
        return np.sqrt(omega_m0*np.exp(-3*self.n_array) + 1 - omega_m0)

    def compute_dimless_age(self, lcdm=False):
        """
        Computes the dimensionless age of the universe H_0 t_0 for the
        Quintessence model(s), or for the LCDM model.
        """
        hubble_parameter = self.hubble_sequential
        if lcdm:
            hubble_parameter = self.hubble_lcdm_sequential
        return simpson(-1/hubble_parameter, self.n_sequential)

    def compute_dimless_lum_dist(self, lcdm=False):
        """
        Compute the dimensionless luminosity distance.
        Valid for both the quintessence models, aswell as the LCDM model.
        Returns:
            (float) luminosity distance.
        """
        hubble_parameter = self.hubble_sequential
        if lcdm:
            hubble_parameter = self.hubble_lcdm_sequential
        integral_flipped = (
            np.exp(-self.n_sequential)*cumulative_trapezoid(
                -np.exp(-self.n_sequential) /
                hubble_parameter, self.n_sequential,
                initial=0)
        )
        return np.flip(integral_flipped)
    
    def chi_squared(self):
        """
        implement the chi squared method to quantify how well the model fits.
        """
        c = 1
        h = 0.7
        lum_dist = self.dimless_lum_dist*c/h
        lum_dist_seq = np.flip(lum_dist)
        ns = redshift_to_n(self.data[:,0])
        lum_dist_interp = interp1d(self.n_sequential,lum_dist_seq)(ns)
        return np.sum((lum_dist_interp - self.data[:,1])**2/self.data[:,2]**2,axis=0)

    def chi_squared_lcdm(self):
        """
        implements chi squared to hunt for the best initial mass density
        parameter for the Lambda CDM model.
        """
        def model(n,omega_m0): return np.sqrt(omega_m0*np.exp(-3*n) + 1 - omega_m0)
        obs_lum_dists = self.data[:,1]
        errs = self.data[:, 2]
        ns = redshift_to_n(self.data[:,0])
        popt, pcov = curve_fit(model,ns,obs_lum_dists)
        omega_m0_best = popt[0]
        lum_dist_model = model(ns,omega_m0_best)
        chi_squared = np.sum((lum_dist_model - obs_lum_dists)**2 / errs**2)
        return popt, pcov, chi_squared


    def pot_label(self):
        """
        potential string used in labels
        """
        pot_label = "power law potential"
        if self.exponential_pot:
            pot_label = "exponential potential"
        return pot_label

    def pot_line(self):
        """
        declares which potential is used by linestyle
        Returns:
            line (string):  linestyle to be used for plot, dashed for exp,
                            whole for inverse power-law.
        """
        line = "-"
        if self.exponential_pot:
            line = "--"
        return line

    def plot_density_parameters(self, fig, ax):
        """
        plot the density parameters for the quintessence field,
        radiation, and mass. Also plot their sum, which must be equal to 1.
        Args:
            fig (matplotlib Figure): The Figure object to use for the plot.
            ax (matplotlib Axes): The Axes object to use for the plot.

        Returns:
            matplotlib Figure: The Figure object used for the plot.
            matplotlib Axes: The Axes object used for the plot.
        """
        linestyle_ = "-"
        if self.exponential_pot:
            linestyle_ = "--"
        arrays_sum = (
            self.omega_phi
            + self.omega_m
            + self.omega_r
        )
        ax.plot(self.redshift, self.omega_phi,
                label=r"$\Omega_\phi$",  linestyle=self.pot_line())
        ax.plot(self.redshift, self.omega_m,
                label=r"$\Omega_m$",  linestyle=self.pot_line())
        ax.plot(self.redshift, self.omega_r,
                label=r"$\Omega_r$",  linestyle=self.pot_line())
        #ax.plot(self.redshift, arrays_sum, linestyle='dashed', color='black')
        return fig, ax

    def plot_equation_of_state(self, fig, ax):
        """
        Plot equation of state for quintessence field.
        Args:
            fig (matplotlib Figure): The Figure object to use for the plot.
            ax (matplotlib Axes): The Axes object to use for the plot.

        Returns:
            matplotlib Figure: The Figure object used for the plot.
            matplotlib Axes: The Axes object used for the plot.
    """
        ax.plot(self.redshift, self.eos, label=self.pot_label(),
                linestyle=self.pot_line())
        return fig, ax

    def plot_hubble_parameter(self, fig, ax):
        """
        Plot the Hubble parameter for the quintessence models as a function
        of redshift.

        Args:   
            fig (matplotlib Figure): The Figure object to use for the plot.
            ax (matplotlib Axes): The Axes object to use for the plot.

        Returns:
            matplotlib Figure: The Figure object used for the plot.
            matplotlib Axes: The Axes object used for the plot.
        """
        ax.plot(self.redshift, self.hubble_parameter,
                label=self.pot_label(), linestyle=self.pot_line())
        return fig, ax

    def plot_hubble_lcdm(self, fig, ax):
        """
        Plot the Hubble parameter for the LCDM model as a function of redshift.
        Args:
            fig (matplotlib Figure): The Figure object to use for the plot.
            ax (matplotlib Axes): The Axes object to use for the plot.

        Returns:
            matplotlib Figure: The Figure object used for the plot.
            matplotlib Axes: The Axes object used for the plot.
        """
        ax.plot(self.redshift, self.hubble_parameter_lcdm,
                label=r"$\Lambda$CDM")
        return fig, ax

    def plot_dimless_lum_dist(self, fig, ax):
        """
        Plot the computed dimensionless luminosity distance for the
        quintessence model.
        Args:
            fig (matplotlib Figure): The Figure object to use for the plot.
            ax (matplotlib Axes): The Axes object to use for the plot.

        Returns:
            matplotlib Figure: The Figure object used for the plot.
            matplotlib Axes: The Axes object used for the plot.        
        """
        ax.plot(self.redshift, self.dimless_lum_dist,
                label=self.pot_label(), linestyle=self.pot_line())
        return fig, ax

    def plot_dimless_lum_dist_lcdm(self, fig, ax):
        """
        Plot the computed dimensionless luminosity distance for the
        LCDM model.
        Args:
            fig (matplotlib Figure): The Figure object to use for the plot.
            ax (matplotlib Axes): The Axes object to use for the plot.
        Returns:
            matplotlib Figure: The Figure object used for the plot.
            matplotlib Axes: The Axes object used for the plot.        
        """
        ax.plot(self.redshift, self.dimless_lum_dist_lcdm,
                label=r"$\Lambda$CDM")
        return fig, ax


power_law_pot = ComputeQuantities(exponential_pot=False, init_x1=5e-5,
                                  init_x2=1e-8, init_x3=0.9999, init_lambda=1e9)


exp_pot = ComputeQuantities(
    exponential_pot=True, init_x1=0, init_x2=5e-13, init_x3=0.9999, init_lambda=1)


def plot_density_parameterss(plp=power_law_pot, expp=exp_pot):
    """
    Plots density parameters for quintessence field, radiation, and matter,
    for the power law potential, and the exponential potential, both in the
    same plot.
    Args:
        plp (object): ComputeQuantities power-law potential object.
        expp (object): ComputeQuantities exponential potential object.
    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    fig, ax = plp.plot_density_parameters(fig, ax)
    fig, ax = expp.plot_density_parameters(fig, ax)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Energy density")
    ax.set_title("Density parameters")
    ax.set_xscale("log")
    ax.invert_xaxis()
    fig.legend()
    ax.yaxis.label.set_size(12)
    fig.savefig("density_parameters.pdf")


def plot_eoss(plp=power_law_pot, expp=exp_pot):
    """
    plots equation of state for both potentials in shared plot.
    Args:
        plp (object): ComputeQuantities power-law potential object.
        expp (object): ComputeQuantities exponential potential object.
    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    fig, ax = plp.plot_equation_of_state(fig, ax)
    fig, ax = expp.plot_equation_of_state(fig, ax)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Equation of state [N/m]")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_title("Equation of state")
    fig.legend()
    ax.yaxis.label.set_size(12)
    fig.savefig("equation_of_state.pdf")


def plot_hubble_parameters(plp=power_law_pot, expp=exp_pot):
    """
    Plot dimensionless hubble parameters for the exponential potential, aswell
    as the inverse power law potential.
    Args:
        plp (object): ComputeQuantities power-law potential object.
        expp (object): ComputeQuantities exponential potential object.
    Returns:
        None.        
    """
    fig, ax = plt.subplots(figsize=(6,6))
    fig, ax = plp.plot_hubble_parameter(fig, ax)
    fig, ax = expp.plot_hubble_parameter(fig, ax)
    fig, ax = expp.plot_hubble_lcdm(fig, ax)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$H/H_0$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_title("Hubble parameter")
    ax.legend()
    ax.yaxis.label.set_size(12)
    fig.savefig("hubble_parameters.pdf")


def plot_dimless_lum_dists(plp=power_law_pot, expp=exp_pot):
    """
    Plot the computed dimensionless luminosity distance for the LCDM model,
    and the exponential potential, and inverse power-law potential for the 
    quintessence model.
    Args:
        plp (object): ComputeQuantities power-law potential object.
        expp (object): ComputeQuantities exponential potential object.
    Returns:
        None.       
    """
    fig, ax = plt.subplots(figsize=(6,6))
    fig, ax = plp.plot_dimless_lum_dist(fig, ax)
    fig, ax = expp.plot_dimless_lum_dist(fig, ax)
    fig, ax = expp.plot_dimless_lum_dist_lcdm(fig, ax)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$H_0 d_L/c$")
    ax.set_xlim(0,2)
    #ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_title("Luminosity distance")
    ax.legend()
    ax.yaxis.label.set_size(12)
    fig.savefig("luminosity_distances.pdf")


def print_dimless_age(plp=power_law_pot, expp=exp_pot):
    """
    prints the computed values for the dimensionless age of the universe
    for the various models.
    """
    print(f"""
Dimensionless age of the universe:
    inverse power law potential: {plp.dimless_age:.4g}
    LCDM: {plp.dimless_age_lcdm:.4g}
    exponential potential:{expp.dimless_age:.4g}
    """)


def print_lum_dist_errors(plp=power_law_pot, expp=exp_pot):
    """
    print computed values for the deimensionless luminosity distances as a
    function of redshift.
    """
    print(f"""
chi squared for luminosity distance power-law potential {plp.chi_squared()}
chi squared for luminosity distance exponential potential {expp.chi_squared()}
    """)

def print_best_omega_m0(plp=power_law_pot):
    best_omega_m0, covariance, chi_squared = plp.chi_squared_lcdm()
    print(f"best omega_m0 = {best_omega_m0}")
    print(f"covariance: {covariance}")
    print(f"chi squared: {chi_squared}")
plot_density_parameterss()
plot_eoss()
plot_hubble_parameters()
plot_dimless_lum_dists()
print_dimless_age()
print_lum_dist_errors()
print_best_omega_m0()
#plt.show()
