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


def initial_conditions_ir_resistance(N_init, params):
  """
  Setup the initial conditions for ir resistance
  """

  init_cv = params['init_cv']
  init_alpha = params['init_alpha']
  init_var = (init_cv*init_alpha)**2
  init_k = init_alpha**2 / init_var
  init_v = (init_alpha**2 / init_k) + init_alpha**2

  init = np.array([N_init, N_init*0.001, 0, init_alpha, init_alpha, init_v, init_v])

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

  params['phi'] = params['lam'] / params['alpha']

  return(params)


def make_trajectory_plot(trajectories, ind, time_val=30, loc="center right"):
  """
  Plot trajectories

  """

  tres1 = trajectories[ind][0]
  tres2 = trajectories[ind][1]
  tres3 = trajectories[ind][2]
  tind = tres1.time < time_val
  fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

  colors = ['#66c2a5','#fc8d62', "#8DA0CB"]
  ax.plot(tres1.time[tind], tres1.N[tind] / tres1.N[0], color=colors[0])
  ax.plot(tres2.time[tind], tres2.N[tind] / tres2.N[0], color=colors[1])
  ax.plot(tres3.time[tind], tres3.N[tind] / tres3.N[0], color=colors[2])

  ax.plot(tres1.time[tind], tres1.I[tind] / tres1.N[0], '--', color=colors[0])
  ax.plot(tres2.time[tind], tres2.I[tind] / tres2.N[0], '--', color=colors[1])
  ax.plot(tres3.time[tind], tres3.I[tind] / tres3.N[0], '--', color=colors[2])
  ax.set_ylim(-0.03, 1.03)
  # ax2.set_ylim(-0.03, 1.03)
  ax.set_xlabel("Time (years)", size=12)
  ax.set_ylabel("Relative population size or\nPrevalence", size=12)

  cols = [colors[0], colors[1], colors[2],'black', 'black']
  ls = ['-', '-', '-', '-', '--']
  labels = ["Tolerance", "Avoidance\nresistance", "Intensity-reduction\nresistance", "Relative abundance", "Prevalence"]
  handles = [plt.Line2D([1], [1], color=cols[i], linestyle=ls[i], label=labels[i]) 
             for i in range(len(cols))]
  ax.legend(handles=handles, loc=loc, frameon=False)
  ax.spines['right'].set_visible(None)
  ax.spines['top'].set_visible(None)
  return(ax)

if __name__ == '__main__':
		
  # Draw parameters from Latin hypercube
  samps = 3000
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
  check_convergence = []

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
    init_ir_resistance = initial_conditions_ir_resistance(N_init, params)

    # Run each model and save the results
    models = [eee.tolerance_model_no_tradeoff_approximation, 
              eee.resistance_model_tradeoff,
              eee.intensity_reduction_resistance_model_no_tradeoff]

    init_vals = [init_tolerance, init_resistance, init_ir_resistance]
    colnames = [['N', 'I', 'Z', 'alpha', 'v'],
                ['N', 'I', 'Z', 'betaN', 'betaI', 'vN', 'vI'],
                ['N', 'I', 'Z', 'alphaN', 'alphaI', 'vN', 'vI']]
    change_points = [True, False, False]  # How to evalute when recovery begins

    # Only run the model if certain conditions are met
    if parasiteR0 > 1 and hostr > 0:

      parasiter = eee.max_eig(N_init, params)
        
      if parasiter > 10**-0.5:# and parasiteR0 < 10:

        # Loop through the three models and simulate
        parasiteR0_vals.append(parasiteR0)
        hostr_vals.append(hostr)
        parasiter_vals.append(parasiter)
        lhc_params_used.append(lhc_params[s, :])
        params_used.append(params)
        Ninit_vals.append(N_init)

        trecov = []
        ttraj = []
        tconv = []
        for m in range(len(models)):

          res, out = odeint(models[m], init_vals[m], time, (params,), full_output=True)
          res = pd.DataFrame(res, columns=colnames[m]).assign(time=time)
          tconv.append(out['message'] == "Integration successful.")

          recovery = eee.recovery_metrics(res.N.values, time, 
                                          percent=percent, 
                                          change_point=change_points[m])

          trecov.append(recovery)
          ttraj.append(res)

        trajectories.append(ttraj)
        recovery_metrics.append(trecov)
        check_convergence.append(np.all(tconv))

  # Unpack results
  parasiter_vals = np.array(parasiter_vals)
  parasiteR0_vals = np.array(parasiteR0_vals)
  hostr_vals = np.array(hostr_vals)

  names = ["tol", "ar", "ir"]
  colnames = ['mag_decline', "decline_stop", 'recovery_time']
  recov_tolerance, recov_resistance, recov_ir_resistance = [pd.DataFrame(np.array(x), 
                                                    columns=[g + "_{0}".format(names[i]) for g in colnames]) 
                                      for i, x in enumerate(zip(*recovery_metrics))]

  ## Figure 3
  cond = ((pd.Series(parasiteR0_vals) < 20) & pd.Series(check_convergence)).values
  diff_df = (recov_tolerance[cond] - recov_resistance[cond]).reset_index(drop=True).dropna()
  recov_concat = pd.concat([recov_tolerance[cond], recov_resistance[cond], recov_ir_resistance[cond]], axis=1)

  varnames = ['decline_stop', "recovery_time", "mag_decline"]
  ax_titles = ['Time until recovery begins', 'Time until recovery stops', 
                "Magnitude of decline"]
  all_recov = []
  all_quant = []
  for i, varname in enumerate(varnames):

    recov_rank = recov_concat.filter(like=varname).dropna().rank(axis=1).melt()
    recov_quant = recov_concat.filter(like=varname).dropna()
    recov_quant = recov_quant.div(recov_quant.min(axis=1), axis="rows").melt().assign(value=lambda x: np.log10(x.value))

    pretty_columns = ["Tolerance", "Intensity-reduction\nresistance", "Avoidance\nresistance"] 
    recov_rank.loc[:, "variable"] = recov_rank.variable.map({varname + "_tol": "Tolerance", 
                                                             varname + "_ar": "Avoidance\nresistance",
                                                             varname + "_ir": "Intensity-reduction\nresistance"})

    recov_quant.loc[:, "variable"] = recov_quant.variable.map({varname + "_tol": "Tolerance", 
                                                             varname + "_ar": "Avoidance\nresistance",
                                                             varname + "_ir": "Intensity-reduction\nresistance"})
    recov_rank.loc[:, "var_type"] = varname
    recov_quant.loc[:, "var_type"] = varname


    all_recov.append(recov_rank)
    all_quant.append(recov_quant)

  all_recov_df = pd.concat(all_recov)
  all_quant_df = pd.concat(all_quant)

  all_quant_df.groupby(['var_type', 'variable']).agg({'value': lambda x: 10**np.median(x)})

  grid = sns.catplot(x="variable", y="value", hue="variable", data=all_quant_df,
                     col="var_type", 
                     order=pretty_columns,
                     # scale=2,
                     kind="box",
                     dodge=False,
                     sharey=False,
                     showfliers=False,
                     palette=["#66c2a5", "#fc8d62", "#8DA0CB"])

  axes = grid.axes.ravel()
  ax_titles = ['Time until recovery begins', 'Time until recovery stops', 
                "Magnitude of decline"]
  ylabels = ["Magnitude larger than minimum", "Time / min(Time)", "% / min(%)"]

  for i, ax in enumerate(axes):
    
    if i == 0:
      ax.set_ylabel(ylabels[i], size=12)

    ax.set_xlabel("")
    ax.set_yticklabels(np.round(10**ax.get_yticks(), decimals=1))
    # ax.set_yticks(range(1, 4))
    ax.set_title(ax_titles[i], size=14)
    ax.tick_params(axis="x", labelsize=12)

    h = 0.9
    if i == 0:
      ax.text(0.1, h, "Fastest", transform=ax.transAxes, fontstyle="italic")
      ax.text(0.78, h, "Slowest", transform=ax.transAxes, fontstyle="italic")
      ax.text(0.43, h, "Medium", transform=ax.transAxes, fontstyle="italic")

    if i == 1:
      ax.text(0.1, h, "Slowest", transform=ax.transAxes, fontstyle="italic")
      ax.text(0.78, h, "Fastest", transform=ax.transAxes, fontstyle="italic")
      ax.text(0.43, h, "Medium", transform=ax.transAxes, fontstyle="italic")

    if i == 2:
      ax.text(0.1, h, "Largest", transform=ax.transAxes, fontstyle="italic")
      ax.text(0.78, h, "Largest", transform=ax.transAxes, fontstyle="italic")
      ax.text(0.43, h, "Smallest", transform=ax.transAxes, fontstyle="italic")

  plt.savefig("../results/resistance_vs_tolerance.pdf", bbox_inches="tight")


  # Figure 3: Make comparison resistance vs. tolerance plots
  ind_long = np.where((pd.Series(parasiteR0_vals) > 20) & 
                      (pd.Series(parasiter_vals) < 25))[0][8]#diff_df.recovery_time.argmin()

  ax1 = make_trajectory_plot(trajectories, ind_long, time_val=18, loc="lower center")
  ax1.set_title(r"$R_0$=" + "{0:.2f}".format(parasiteR0_vals[ind_long]))
  print(params_used[ind_long])
  # plt.savefig("../results/long_recovery_for_resistance.pdf", bbox_inches="tight")

  ind_typical = np.where(recov_tolerance.mag_decline_tol.values > 0.5)[0][2]
  ax2 = make_trajectory_plot(trajectories, ind_typical, time_val=30, loc=None)
  ax2.set_title(r"$R_0$=" + "{0:.2f}".format(parasiteR0_vals[ind_typical]))
  print(params_used[ind_typical])
  # plt.savefig("../results/short_recovery_for_tolerance.pdf", bbox_inches="tight")


