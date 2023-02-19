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
or tolerance to a pathogen.  This scripts uses the full E3 model and tests 
how recovery dynamics differ between hosts using resistance and tolerance 
strategies.
"""

def initial_conditions(params, model_type, param_range):
  """
  Setup the initial conditions for full model

  Parameters
  ----------
  params : dict
    Parameter dictionary for full model eee.full_resistance_model or 
    eee.full_tolerance_model
  model_type : str
    Either 'alpha' or 'beta'
  """

  # Set up the initial gamma distribution
  init_var = (params['init_cv']*params[model_type])**2 
  theta = init_var / params[model_type]
  a = params[model_type] / theta
  gamma_dist = stats.gamma(a, scale=theta)

  # Set-up param vals
  p_vals = np.linspace(param_range[model_type][0], param_range[model_type][1], 
                       num=params['bins'])
  Δ = p_vals[1] - p_vals[0]
  probs = gamma_dist.cdf(p_vals + Δ) - gamma_dist.cdf(p_vals)
  probs_norm = probs / np.sum(probs)
  p_vals_mid = p_vals + (Δ / 2)
  params['{0}_vals'.format(model_type)] = p_vals_mid
  params['r_vals'] = eee.r_fxn(p_vals_mid, 1e-10, params['r'])

  # Set-up the full model simulation
  N_init = (params['r'] - params['mu']) / params['delta']
  Nvals = probs_norm * N_init
  Ivals = np.zeros(len(Nvals))
  init_vals = np.concatenate([Nvals, Ivals, [1e-5]])

  return(init_vals)


def set_params(s, plist, lhc_params):
  """
  Set up parameter dictionary
  """

  params = {}
  params['beta_m'] = 0
  params['alpha_m'] = 0
  for i, p in enumerate(plist):

    params[p] = lhc_params[s, i]

    # Set alpha and beta 
    if p == 'init_beta':
      params['beta'] = params['init_beta']

    if p == 'init_alpha':
      params['alpha'] = params['init_alpha']

  return(params)


def make_trajectory_plot(trajectories, ind, bins, time_val=30):
  """
  Plot trajectories
  """

  tres1 = trajectories[ind][0]
  tres2 = trajectories[ind][1]
  tind = tres1.time < time_val
  fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

  colors = ['#66c2a5','#fc8d62']
  ax.plot(tres1.time[tind], tres1.N.values[tind] / tres1.N.values[0], color=colors[0])
  ax.plot(tres2.time[tind], tres2.N.values[tind] / tres2.N.values[0], color=colors[1])

  ax.plot(tres1.time[tind], tres1.I.values[tind] / tres1.N.values[tind], '--', color=colors[0])
  ax.plot(tres2.time[tind], tres2.I.values[tind] / tres2.N.values[tind], '--', color=colors[1])
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
  samps = 500
  bins = 200 # Discrete number of bins to approximate ODE
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
  
  # Parameter range of alpha and beta   
  param_range = {'alpha': (0, 10), 'beta': (0, 2)}

  for s in range(samps):

    if s % 2 == 0:
      print(s + 1)
	  # Set parameters for simulation
    params = set_params(s, plist, lhc_params)
    params['bins'] = bins

    # Prepare initial values for simulation
    N_init = (params['r'] - params['mu']) / params['delta']
    parasiteR0 = (N_init*params['beta']*params['lam']) / ((params['alpha'] + params['mu'])*params['mu_z'])
    hostr = params['r'] - params['mu']

    init_tolerance = initial_conditions(params, 'alpha', param_range)
    init_resistance = initial_conditions(params, 'beta', param_range)

    # Run each model and save the results
    models = [eee.full_tolerance_model, 
              eee.full_resistance_model]

    init_vals = [init_tolerance, init_resistance]
    change_points = [False, False]  # How to evalute when recovery begins

    # Only run the model if certain conditions are met
    if parasiteR0 > 1 and hostr > 0:

      parasiter = eee.max_eig(N_init, params)
        
      if parasiter > 10**-0.5:

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

          res = pd.DataFrame(odeint(models[m], init_vals[m], time, (params,))).assign(time=time)
          full_N = res.iloc[:, :params['bins']].sum(axis=1)
          full_I = res.iloc[:, params['bins']:(params['bins']*2)].sum(axis=1)
          res_small = pd.DataFrame(dict(time=res.time.values, N=full_N, I=full_I))

          recovery = eee.recovery_metrics(full_N, time, 
                                          percent=percent, 
                                          change_point=change_points[m])

          trecov.append(recovery)
          ttraj.append(res_small)

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


  # Explore the trajectories from the full model to ensure
  # that we are calculating the correct time of declines
  colors = list(sns.color_palette(n_colors=len(trajectories)).as_hex())
  for t in np.arange(len(trajectories))[::10]:

    tol_traj, res_traj = trajectories[t]

    ttime = tol_traj.time.values
    time_ind = ttime < 50

    # Plot resistance trajectories and metrics
    plt.plot(res_traj.time[time_ind], res_traj.N[time_ind], '-', color=colors[t])
    plt.vlines(recov_resistance.decline_stop[t], np.min(res_traj.N), np.max(res_traj.N), ls='--', color=colors[t])
    plt.vlines(recov_resistance.decline_stop[t] + recov_resistance.recovery_time[t], np.min(res_traj.N), np.max(res_traj.N), ls='-', color=colors[t])

    # Plot tolerance trajectories and metrics
    plt.plot(tol_traj.time[time_ind], tol_traj.N[time_ind], '--', color = colors[t])
    plt.vlines(recov_tolerance.decline_stop[t], np.min(tol_traj.N), np.max(tol_traj.N), ls='--', color=colors[t])
    plt.vlines(recov_tolerance.decline_stop[t] + recov_tolerance.recovery_time[t], np.min(tol_traj.N), np.max(tol_traj.N), ls='-', color=colors[t])


  ## Figure S3

  cond = parasiteR0_vals < 20
  diff_df = (recov_tolerance[cond] - recov_resistance[cond]).dropna().reset_index(drop=True)
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

  # plt.savefig("../results/resistance_vs_tolerance_full_model.pdf", bbox_inches="tight")

  ## Figure S3: Make comparison resistance vs. tolerance plots
  ind_long = np.where(parasiteR0_vals > 20)[0][1]#diff_df.recovery_time.argmin()
  ax1 = make_trajectory_plot(trajectories, ind_long, bins, time_val=10)
  ax1.set_title(r"$R_0$=" + "{0:.2f}".format(parasiteR0_vals[ind_long]))
  # plt.savefig("../results/long_recovery_for_resistance_full_model.pdf", bbox_inches="tight")

  ind_typical = np.where(recov_tolerance.mag_decline.values > 0.5)[0][10]
  ax2 = make_trajectory_plot(trajectories, ind_typical, bins, time_val=30)
  ax2.set_title(r"$R_0$=" + "{0:.2f}".format(parasiteR0_vals[ind_typical]))
  # plt.savefig("../results/short_recovery_for_tolerance_full_model.pdf", bbox_inches="tight")


