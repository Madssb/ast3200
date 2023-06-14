
from astropy import constants as const
import matplotlib.pyplot as plt
# for zooming in in plot
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np

REDUCED_PLANCK_CONST = const.hbar
SPEED_OF_LIGHT = const.c
GRAV_CONST = const.G
PLANCK_ENERGY = np.sqrt(REDUCED_PLANCK_CONST*SPEED_OF_LIGHT**5/GRAV_CONST)
PLANCK_MASS = np.sqrt(REDUCED_PLANCK_CONST*SPEED_OF_LIGHT/GRAV_CONST)
PLANCK_LENGTH = np.sqrt(REDUCED_PLANCK_CONST*GRAV_CONST/SPEED_OF_LIGHT**3)


def v_phi_squared(psi, psi_init):
  """
  Compute the dimensionless potential for the phi squared inflation model.
  """
  return 3/(8*np.pi)*(psi/psi_init)**2


def v_diff_psi_phi_squared(psi, psi_init):
  """
  Compute the differentiation of  the dimensionless potential with respect to
  phi for the phi squared inflation model.  
  """
  return 3*psi/(4*np.pi*psi_init**2)


def v_starobinsky(psi, psi_init):
  """
  Compute the dimensionless potential for the Starobinsky model
  """
  return (3/(8*np.pi)*(1 - np.exp(-np.sqrt(16*np.pi/3)*psi))**2
          / (np.exp(-np.sqrt(16*np.pi/3)*psi_init)**2)**2)


def v_diff_psi_starobinsky(psi, psi_init):
  """
  Compute the differentiation of  the dimensionless potential with respect to
  phi for the Starobinsky model.
  """
  return (3/(4*np.pi)/(np.exp(-np.sqrt(16*np.pi/3)*psi_init)**2)**2
          * np.sqrt(16*np.pi/3)*(1 - np.exp(-np.sqrt(16*np.pi/3)*psi))
          * np.exp(-np.sqrt(16*np.pi/3)*psi))


class InflationSim:
  """
  Solves psi for specified inflation model.
  """
  def __init__(self, psi_init=8.925, psi_diff_tau_init=0, tau_end=1400,
               phi_squared_model_used=True):
    """
    Instantiates object of InflationSim.

    Args:
      v (Callable): Unitless potential
      v_diff_psi (Callable): derivative of unitless potential with respect to psi.
      psi_init (Float): initial value for psi. varies with model used.
      tau_end (Float): duration of simulation in tau.
    """
    self.N = 10000
    self.psis = np.empty(self.N)
    self.psis[0] = psi_init
    self.psis_diff_tau = np.empty(self.N)
    self.psis_diff_tau[0] = psi_diff_tau_init
    self.psi_init = psi_init
    self.TAUS = np.linspace(0, tau_end, self.N)
    self.TAU_DIFF = tau_end/self.N
    


    # problem e
    self.compute_psi()
    if phi_squared_model_used:
      self.v = lambda psi: 3/(8*np.pi)*(psi/self.psi_init)**2
      self.v_diff_psi = lambda psi: 3*psi/(4*np.pi*psi_init**2)
      self.slow_roll_parameter = lambda psi: 1/(4*np.pi*self.psis**2)
    else:
      self.v = (
        lambda psi: (3/(8*np.pi)*(1 - np.exp(-np.sqrt(16*np.pi/3)*psi))**2
          / (np.exp(-np.sqrt(16*np.pi/3)*psi_init)**2)**2))
      self.v_diff = (
        lambda psi: (3/(4*np.pi)/(np.exp(-np.sqrt(16*np.pi/3)*psi_init)**2)**2
          * np.sqrt(16*np.pi/3)*(1 - np.exp(-np.sqrt(16*np.pi/3)*psi))
          * np.exp(-np.sqrt(16*np.pi/3)*psi)) 
      )

    self.compute_dimless_hubble_param()
    self.compute_slow_roll_approximation_psi()
    self.compute_slow_roll_parameter()
    self.compute_total_e_folds()
    self.compute_ratio()
    self.compute_e_folds()

  def compute_psi(self):
    """
    Compute psi and the differential of psi with respect to tau, as functions
    of tau.
    """
    h = 1
    for i in range(self.N-1):
      psi_diff_tau_2nd_order = (-3 * h * self.psis_diff_tau[i]
                                - self.v_diff_psi(self.psis[i]))
      self.psis_diff_tau[i+1] = (self.psis_diff_tau[i]
                                 + psi_diff_tau_2nd_order*self.TAU_DIFF)
      self.psis[i+1] = self.psis[i] + self.psis_diff_tau[i+1]*self.TAU_DIFF
      h = np.sqrt(
          8*np.pi/3*(self.psis_diff_tau[i+1]**2/2 + self.v(self.psis[i+1])))

  def compute_dimless_hubble_param(self):
    """
    Compute dimensionless hubble parameter as a function of psi. 
    """
    self.dimless_hubble = 8*np.pi/3 * \
        (self.psis_diff_tau**2/2 + self.v(self.psis))

  def compute_slow_roll_approximation_psi(self):
    """
    Compute the slow-roll approximated psi as a function of tau.
    """
    self.slow_roll_psi = self.psi_init - self.TAUS/(4*np.pi*self.psi_init)

  def compute_slow_roll_parameter(self):
    """
    Compute the slow-roll parameter as a function of tau.
    """
    self.slow_roll_parameter = 1/(4*np.pi*self.psis**2)

  def compute_total_e_folds(self):
    """
    Compute the total number of e-folds for inflation simulation.
    """
    for i, parameter in enumerate(self.slow_roll_parameter):
      if np.abs(parameter - 1) < 1e-2:
        self.psi_end = self.psis[i]
        self.tau_end = self.TAUS[i]
        break
    try:
      self.e_folds_tot = 2*np.pi*(self.psi_init**2 - self.psi_end**2)
    except AttributeError:
      print("")

  def compute_ratio(self):
    """
    Compute the one ratio in idk.
    """
    self.ratio = (self.psis_diff_tau**2/2 + self.v(self.psis)
                  / (self.psis_diff_tau**2/2 - self.v(self.psis)))

  def compute_e_folds(self):
    """
    Compute the e-folds as a function of psi.
    """
    self.e_folds = self.e_folds_tot - 2*np.pi*(self.psi_init**2 - self.psis)

  def compute_and_plot_nr(self):
    e_folds = np.linspace(50, 60, 10000)
    slow_roll_parameter = 1 / \
        (2*(e_folds - self.e_folds_tot) + 4*np.pi*self.psi_init**2)

    n = 1 - 8*slow_roll_parameter + 2*slow_roll_parameter
    r = 16*slow_roll_parameter
    fig, ax = plt.subplots()
    ax.plot(n, r)
    ax.set_xlabel("n")
    ax.set_ylabel("r")
    return fig
    fig.savefig("k.pdf")

  def plot_psi(self):
    """
    IDK
    """
    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(self.TAUS, self.psis/self.psi_init)
    ax[1].plot(self.TAUS, self.dimless_hubble)
    ax[2].plot(self.TAUS, self.dimless_hubble**2)

    ax[0].set_ylabel(r"$\psi / \psi_i$")
    ax[1].set_ylabel(r"$h$")
    ax[2].set_ylabel(r"$h^2$")

    ax[-1].set_xlabel(r"$\tau$")
    ax[-1].set_xlim(0, self.tau_end*1.3)
    ax[-1].set_ylim(0, 1)

    # Create a zoomed-in inset axes for ax[0]
    axins = zoomed_inset_axes(ax[0], zoom=10, loc="upper right")
    axins.plot(self.TAUS, self.psis/self.psi_init)
    axins.plot(self.TAUS, np.zeros_like(self.TAUS),
               color="black", linestyle="dashed")
    # Set x-axis and y-axis limits for the zoomed-in inset axes
    axins.set_xlim(self.tau_end*1.072, self.tau_end*1.122)
    axins.set_ylim(-0.01, 0.01)
    # Remove the ticks and tick labels on the zoomed-in inset axes
    axins.set_xticks([])
    axins.set_yticks([])
    # Mark the region of the original plot that corresponds to the zoomed-in inset axes
    mark_inset(ax[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # Return the figure and axes objects
    return fig
    fig.savefig("e.pdf")

  def plot_sra(self):
    """
    Visualize slow roll approximation along with numerical solution.
    """
    fig, ax = plt.subplots()
    ax.plot(self.TAUS, self.psis/self.psi_init, label="numerical")
    ax.plot(self.TAUS, self.slow_roll_psi/self.psi_init,
            label="SRA", linestyle="dashed")

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\psi / \psi_i$")

    ax.set_xlim(0, self.tau_end*1.3)
    ax.set_ylim(-1, 1)
    fig.legend()
    return fig
    fig.savefig("f.pdf")

  def plot_slow_roll_parameter_psi(self):
    """
    Visualize slow roll parameter (epsilon).
    """
    fig, ax = plt.subplots()
    ax.plot(self.TAUS, self.slow_roll_parameter)

    ax.plot(self.TAUS, np.ones(self.N), color="black",
            linestyle="dashed", linewidth=0.5)
    ax.plot(np.ones(self.N)*self.tau_end, np.linspace(0, 2, self.N),
            color="black", linestyle="dashed", linewidth=0.5)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("slow roll parameter e")

    ax.set_xlim(0, self.tau_end*1.3)
    ax.set_ylim(0, 2)
    return fig
    fig.savefig("g.pdf")

  def plot_ratio(self):
    """
    Visualize ratio
    """
    fig, ax = plt.subplots()
    ax.plot(self.TAUS, self.ratio)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\frac{p_\phi}{\rho_\phi c^2}$")

    ax.set_xlim(0, self.tau_end*1.3)
    ax.set_ylim(-2, 2)
    return fig
    fig.savefig("i.pdf")

  def plot_slow_roll_parameter_e_folds(self):
    """
    Visualize slow roll parameter as function of e-folds.
    """
    fig, ax = plt.subplots()
    ax.plot(self.e_folds, self.slow_roll_parameter)
    ax.set_xlabel("number of e-folds N")
    ax.set_ylabel(r"slow-roll parameter e and $\eta$")
    ax.set_yscale("log")
    ax.invert_xaxis()  # Invert the x-axis
    return fig
    fig.savefig("j.pdf")


def phi_squared_model():
  """
  Compute and plot all quantities of interest for phi squared model.
  """
  phi_squared_model = InflationSim(v=v_phi_squared, v_diff_psi=v_diff_psi_phi_squared)
  
  fig_plot_psi = phi_squared_model.plot_psi()
  fig_plot_psi.savefig("e.pdf")
  
  fig_plot_sra = phi_squared_model.plot_sra()
  fig_plot_sra.savefig("f.pdf")
  
  fig_plot_slow_roll_parameter_psi = phi_squared_model.plot_slow_roll_parameter_psi()
  fig_plot_slow_roll_parameter_psi.savefig("g.pdf")
  
  fig_plot_ratio = phi_squared_model.plot_ratio()
  fig_plot_ratio.savefig("i.pdf")
  
  fig_plot_slow_roll_parameter_e_folds = phi_squared_model.plot_slow_roll_parameter_e_folds()
  fig_plot_slow_roll_parameter_e_folds.savefig("j.pdf")

  fig_compute_and_plot_nr = phi_squared_model.compute_and_plot_nr()
  fig_compute_and_plot_nr.savefig("k.pdf")


def starobinsky_model():
  starobinsky_model = InflationSim(v=v_starobinsky, v_diff_psi=v_diff_psi_starobinsky, psi_init=2)
  fig = starobinsky_model.plot_psi()
  fig.savefig("m.pdf")


if __name__ == "__main__":
  #phi_squared_model()
  starobinsky_model()
