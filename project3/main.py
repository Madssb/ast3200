
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
  return (
      3 / (8*np.pi)
      * (1 - np.exp(-np.sqrt(16*np.pi/3)*psi))**2
      / (1 - np.exp(-np.sqrt(16*np.pi/3)*psi_init))**2
  )


def v_diff_psi_starobinsky(psi, psi_init):
  """
  Compute the differentiation of  the dimensionless potential with respect to
  phi for the Starobinsky model.
  """
  return (
      3 / (4*np.pi)
      / (1 - np.exp(-np.sqrt(16*np.pi/3)*psi_init))**2
      * np.sqrt(16*np.pi/3)
      *(1 - np.exp(-np.sqrt(16*np.pi/3)*psi))
      * np.exp(-np.sqrt(16*np.pi/3)*psi)
  )


class InflationSim:
  """
  Solves psi for specified inflation model.
  """

  def __init__(self, phi_squared_model_used = True, psi_init=8.925,
               psi_diff_tau_init=0, tau_end=1400):
    """
    Instantiates object of InflationSim.

    Args:
      phi_squared_model_used (Boolean): True if phi square model used.
      psi_init (Float): initial value for psi. varies with model used.
      psi_diff_tau_init (Float): Initial value for psi diff tau.
      tau_end (Float): duration of simulation in tau.
    """
    self.N = 1000000
    self.psis = np.empty(self.N)
    self.psis_diff_tau = np.empty(self.N)
    self.ln = np.empty(self.N)
    self.dimless_hubble = np.empty(self.N)
    self.psis[0] = psi_init
    self.psis_diff_tau[0] = psi_diff_tau_init
    self.ln[0] = 0
    self.dimless_hubble[0] = 1
    self.psi_init = psi_init
    self.TAUS = np.linspace(0, tau_end, self.N)
    self.TAU_DIFF = tau_end/self.N
    self.phi_squared_model_used = phi_squared_model_used
    if phi_squared_model_used:
      self.v = lambda psi: v_phi_squared(psi, psi_init)
      self.v_diff_psi = lambda psi: v_diff_psi_phi_squared(psi, psi_init)
    else:
      self.v = lambda psi: v_starobinsky(psi, psi_init)
      self.v_diff_psi = lambda psi: v_diff_psi_starobinsky(psi, psi_init)
    # problem e
    self.compute_psi()
    self.slow_roll_psi = self.psi_init - self.TAUS/(4*np.pi*self.psi_init)
    self.compute_slow_roll_parameters()
    self.compute_total_e_folds()
    end_index = np.where(self.slow_roll_parameter <= 1)[0][-1]
    self.e_folds_tot = self.ln[end_index]
    self.psi_end = self.psis[end_index]
    self.tau_end = self.TAUS[end_index]
    self.ratio = (self.psis_diff_tau**2/2 + self.v(self.psis)
                  / (self.psis_diff_tau**2/2 - self.v(self.psis)))
    self.e_folds = self.e_folds_tot - self.ln

  def compute_psi(self):
    """
    Compute psi and the differential of psi with respect to tau, as functions
    of tau.
    """
    for i in range(self.N-1):
      psi_diff_tau_2nd_order = (-3 * self.dimless_hubble[i] * self.psis_diff_tau[i]
                                - self.v_diff_psi(self.psis[i]))
      self.psis_diff_tau[i+1] = (self.psis_diff_tau[i]
                                 + psi_diff_tau_2nd_order*self.TAU_DIFF)
      self.psis[i+1] = self.psis[i] + self.psis_diff_tau[i+1]*self.TAU_DIFF
      self.dimless_hubble[i+1] = np.sqrt(
          8*np.pi/3*(self.psis_diff_tau[i+1]**2/2 + self.v(self.psis[i+1])))    
      self.ln[i+1] = self.ln[i] + self.dimless_hubble[i+1]*self.TAU_DIFF

  def compute_slow_roll_parameters(self):
    """
    Compute the slow-roll parameter as a function of tau.
    """
    if self.phi_squared_model_used:
      self.slow_roll_parameter = 1/(4*np.pi*self.psis**2)
    else:
      y = -np.sqrt(16*np.pi/3)*self.psis
      self.slow_roll_parameter = 4/3*np.exp(2*y)/(1 - np.exp(y))**2
      self.slow_roll_parameter_2 = 4/3*(2*np.exp(2*y) - np.exp(y))/(1 - np.exp(y))**2

  def compute_total_e_folds(self):
    """
    Compute the total number of e-folds for inflation simulation.
    """
    # self.e_folds_tot = self.ln[]
    # for i, parameter in enumerate(self.slow_roll_parameter):
    #   if np.abs(parameter - 1) < 1e-2:
    #     self.psi_end = self.psis[i]
    #     self.tau_end = self.TAUS[i]
    #     break
    # self.e_folds_tot = 2*np.pi*(self.psi_init**2 - self.psi_end**2)


  def compute_and_plot_nr(self):
    e_folds = np.linspace(50, 60, 10000)
    slow_roll_parameter = (
      1 / (2*(e_folds - self.e_folds_tot) + 4*np.pi*self.psi_init**2))

    n = 1 - 8*slow_roll_parameter + 2*slow_roll_parameter
    r = 16*slow_roll_parameter
    fig, ax = plt.subplots()
    ax.plot(n, r)
    ax.set_xlabel("n")
    ax.set_ylabel("r")
    return fig

  def plot_psi(self):
    """
    Visualize psi as a function of tau.
    """
    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(self.TAUS, self.psis/self.psi_init)
    ax[1].plot(self.TAUS, self.ln)
    ax[2].plot(self.TAUS, self.dimless_hubble**2)

    ax[0].set_ylabel(r"$\psi / \psi_i$")
    ax[1].set_ylabel(r"$ln(a/a_i)$")
    ax[2].set_ylabel(r"$h^2$")

    ax[-1].set_xlabel(r"$\tau$")
    ax[-1].set_xlim(0, self.tau_end*1.3)
    ax[-1].set_ylim(0, 1)
    if self.phi_squared_model_used:
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

  def plot_slow_roll_parameter_psi(self):
    """
    Visualize slow roll parameter (epsilon) as a function of psi.
    """
    fig, ax = plt.subplots()
    ax.plot(self.TAUS, self.slow_roll_parameter)

    ax.plot(self.TAUS, np.ones(self.N), color="black",
            linestyle="dashed", linewidth=0.5)
    ax.plot(np.ones(self.N)*self.tau_end, np.linspace(0, 2, self.N),
            color="black", linestyle="dashed", linewidth=0.5)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("slow roll parameter $\epsilon$")

    ax.set_xlim(0, self.tau_end*1.3)
    ax.set_ylim(0, 2)
    return fig

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

  def plot_slow_roll_parameter_e_folds(self):
    """
    Visualize slow roll parameter as function of e-folds.
    """
    fig, ax = plt.subplots()
    ax.plot(self.e_folds[self.slow_roll_parameter<1], self.slow_roll_parameter[self.slow_roll_parameter<1], label=r"$\epsilon$")
    if not self.phi_squared_model_used:
      ax.plot(self.e_folds[self.slow_roll_parameter<1], self.slow_roll_parameter_2[self.slow_roll_parameter<1], label = r"$\eta$", linestyle="dashed")
      fig.legend()
    ax.set_xlabel("number of e-folds N")
    ax.set_ylabel(r"slow-roll parameter $\epsilon$ and $\eta$")
    #ax.set_yscale("log")
    ax.set_xscale("log")
    ax.invert_xaxis()  # Invert the x-axis
    return fig


def phi_squared_model():
  """
  Compute and plot all quantities of interest for phi squared model.
  """
  phi_squared_model = InflationSim()
  
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
  print(f"e-folds tot: {phi_squared_model.e_folds_tot:.4g}")


def starobinsky_model():
  starobinsky_model = InflationSim(phi_squared_model_used=False, psi_init=2, tau_end=10000)
  fig = starobinsky_model.plot_psi()
  fig.savefig("m.pdf")

  fig = starobinsky_model.plot_slow_roll_parameter_e_folds()
  fig.savefig("n_1.pdf")

  fig = starobinsky_model.compute_and_plot_nr()
  fig.savefig("n_2.pdf")
  print(f"e-folds tot: {starobinsky_model.e_folds_tot:.4g}")





if __name__ == "__main__":
  phi_squared_model()
  starobinsky_model()
  plt.show()
