from scipy.integrate import odeint
import scipy.optimize as opt
import numpy as np
import scipy.stats as stats
from scipy.stats import qmc
import pandas as pd
import matplotlib.pyplot as plt
import epiecoevo_functions as eee
import importlib
import seaborn as sns
from sympy import symbols, Matrix, solve, lambdify
importlib.reload(eee)

"""
Script to analyze and compare recovery curves from hosts evolving resistance
or tolerance to a pathogen.  This scripts uses the reduced/moment-closure 
E3 model and tests how recovery dynamics differ between hosts using resistance 
and tolerance strategies.
"""

def initial_conditions_tolerance(N_init, params):
  """
  Setup the initial conditions for the tolerance simulation
  """

  init_cv = params['init_cv']
  init_alpha = params['init_alpha']
  init_var = (init_cv*init_alpha)**2
  init_k = init_alpha**2 / init_var
  init_v = (init_alpha**2 / init_k) + init_alpha**2

  init = np.array([N_init, N_init*0.001, 0, init_alpha, init_v])

  return(init)


def initial_conditions_resistance(N_init, params):
  """
  Setup the initial conditions for the resistance simulation
  """

  # Set up initial values
  init_cv = params['init_cv']
  init_beta = params['init_beta']
  init_var = (init_cv*init_beta)**2
  init_k = init_beta**2 / init_var
  init_v = (init_beta**2 / init_k) + init_beta**2

  init = np.array([N_init, N_init*0.001, 0, init_beta, init_beta, init_v, 
                   init_v])
  return(init)


def set_params(s, plist, lhc_params):
  """
  Set up parameter dictionary
  """

  params = {}
  params['beta_m'] = 0.0
  for i, p in enumerate(plist):

    params[p] = lhc_params[s, i]

    # Set alpha and beta 
    if p == 'init_beta':
      params['beta'] = params['init_beta']

    if p == 'init_alpha':
      params['alpha'] = params['init_alpha']

  return(params)


def make_trajectory_plot(trajectories, ind, time_val=30):
  """
  Plot trajectories

  """

  tres1 = trajectories[ind][0]
  tres2 = trajectories[ind][1]
  tind = tres1.time < time_val
  fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

  colors = ['#66c2a5','#fc8d62']
  ax.plot(tres1.time[tind], tres1.N[tind] / tres1.N[0], color=colors[0])
  ax.plot(tres2.time[tind], tres2.N[tind] / tres2.N[0], color=colors[1])

  ax.plot(tres1.time[tind], tres1.I[tind] / tres1.N[0], '--', color=colors[0])
  ax.plot(tres2.time[tind], tres2.I[tind] / tres2.N[0], '--', color=colors[1])
  ax.set_ylim(-0.03, 1.03)
  # ax2.set_ylim(-0.03, 1.03)
  ax.set_xlabel("Time (years)", size=12)
  ax.set_ylabel("Relative population size or\nPrevalence", size=12)

  cols = [colors[0], colors[1], 'black', 'black']
  ls = ['-', '-', '-', '--']
  labels = ["Tolerance", "Resistance", "Relative abundance", "Prevalence"]
  handles = [plt.Line2D([1], [1], color=cols[i], linestyle=ls[i], label=labels[i]) 
             for i in range(len(cols))]
  ax.legend(handles=handles, loc="center right", frameon=False)
  ax.spines['right'].set_visible(None)
  ax.spines['top'].set_visible(None)
  return(ax)

if __name__ == '__main__':
		
  # Draw parameters from Latin hypercube
  samps = 1000
  plist = ['init_beta', 'lam', 'mu_z', 'r', 'mu', 'init_cv', 'init_alpha', 'delta']
  lower = [0.05, 100,  10,    0.1, 0.1, 0.5,  1, 0.1]
  upper = [0.3,  1000, 100,   10,  3,   1.5,   3, 0.5]
  sampler = qmc.LatinHypercube(d=len(lower))
  sample = sampler.random(samps)
  lhc_params = qmc.scale(sample, lower, upper)

  # Simulate time
  time = np.linspace(0, 5000, num=50000)
  percent = 0.02 # Recovery occurs within 2% of original value
  np.random.seed(30)

  # Predictors
  parasiteR0_vals = []
  parasiter_vals = []
  hostr_vals = []
  lhc_params_used = []
  params_used = []
  Ninit_vals = []

  # Output 
  trajectories = []
  recovery_metrics = []
  time_decline_stop = []
  time_full_recovery = []
  magnitude_of_decline = []

  for s in range(samps):

    if s % 10 == 0:
      print(s + 1)
	  # Set parameters for simulation
    params = set_params(s, plist, lhc_params)

    # Prepare initial values for simulation

    N_init = (params['r'] - params['mu']) / params['delta']
    parasiteR0 = (N_init*params['beta']*params['lam']) / ((params['alpha'] + params['mu'])*params['mu_z'])
    hostr = params['r'] - params['mu']

    init_tolerance = initial_conditions_tolerance(N_init, params)
    init_resistance = initial_conditions_resistance(N_init, params)

    # Run each model and save the results
    models = [eee.tolerance_model_no_tradeoff_approximation, 
              eee.resistance_model_tradeoff]

    init_vals = [init_tolerance, init_resistance]
    colnames = [['N', 'I', 'Z', 'alpha', 'v'],
                ['N', 'I', 'Z', 'betaN', 'betaI', 'vN', 'vI']]
    change_points = [True, False]  # How to evalute when recovery begins

    # Only run the model if certain conditions are met
    if parasiteR0 > 1 and hostr > 0:

      parasiter = eee.max_eig(N_init, params)
        
      if parasiter > 10**-0.5:# and parasiteR0 < 10:

        # Loop through the two models and simulate
        parasiteR0_vals.append(parasiteR0)
        hostr_vals.append(hostr)
        parasiter_vals.append(parasiter)
        lhc_params_used.append(lhc_params[s, :])
        params_used.append(params)
        Ninit_vals.append(N_init)

        trecov = []
        ttraj = []
        for m in range(len(models)):

          res = pd.DataFrame(odeint(models[m], init_vals[m], time, (params,)),
                                    columns=colnames[m]).assign(time=time)

          recovery = eee.recovery_metrics(res.N.values, time, 
                                          percent=percent, 
                                          change_point=change_points[m])

          trecov.append(recovery)
          ttraj.append(res)

        trajectories.append(ttraj)
        recovery_metrics.append(trecov)

  # Unpack results
  parasiter_vals = np.array(parasiter_vals)
  parasiteR0_vals = np.array(parasiteR0_vals)
  hostr_vals = np.array(hostr_vals)
  recov_tolerance, recov_resistance = [pd.DataFrame(np.array(x), 
                                                    columns=['mag_decline', 
                                                              "decline_stop",
                                                              'recovery_time']) 
                                      for x in zip(*recovery_metrics)]

  ## Figure 3
  cond = parasiteR0_vals < 20
  diff_df = (recov_tolerance[cond] - recov_resistance[cond]).reset_index(drop=True)
  plt.boxplot(diff_df.mag_decline)
  melted_diff = diff_df.melt().assign(metric="metric")
  col_order = ['decline_stop', 'recovery_time', 'mag_decline']
  grid = sns.catplot(x="metric", y="value", col="variable", hue="variable", 
              col_order=col_order,
              hue_order=col_order,
              sharey=False, data=melted_diff, zorder=-10,
              aspect=0.6, height=4, sharex=False, size=1.5)
  ax_titles = ['Time until\nrecovery begins', 'Time until\nrecovery stops', 
                "Magnitude of\ndecline"]
  ylabels = ["(time)", "(time)", "(%)"]
  axes = grid.axes.ravel()

  for i, ax in enumerate(axes):
    sns.boxenplot(y=col_order[i], data=diff_df, ax=ax, width=0.2, 
                   color=sns.color_palette()[i], showfliers=False)
    ax.set_xlabel(None)
    ax.set_ylabel("Tolerance - resistance " + ylabels[i])
    xlim = ax.get_xlim()
    ax.hlines(0, *xlim, color='red')
    ax.set_xlim(xlim)
    ax.set_xticklabels([ax_titles[i]])
    ax.set_title("")

  # plt.savefig("../results/resistance_vs_tolerance.pdf", bbox_inches="tight")

  ## Figure 3: Make comparison resistance vs. tolerance plots
  ind_long = np.where(parasiteR0_vals > 20)[0][0]#diff_df.recovery_time.argmin()
  ax1 = make_trajectory_plot(trajectories, ind_long, time_val=10)
  ax1.set_title(r"$R_0$=" + "{0:.2f}".format(parasiteR0_vals[ind_long]))
  # plt.savefig("../results/long_recovery_for_resistance.pdf", bbox_inches="tight")

  ind_typical = np.where(recov_tolerance.mag_decline.values > 0.5)[0][4]
  ax2 = make_trajectory_plot(trajectories, ind_typical, time_val=30)
  ax2.set_title(r"$R_0$=" + "{0:.2f}".format(parasiteR0_vals[ind_typical]))
  # plt.savefig("../results/short_recovery_for_tolerance.pdf", bbox_inches="tight")


