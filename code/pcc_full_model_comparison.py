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
from statsmodels.api import GLM
from patsy import dmatrix, dmatrices
from pingouin import partial_corr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
importlib.reload(eee)

plt.rcParams['font.sans-serif'] = "Arial"

"""
Perform the same PCC analysis as Fig. 2 in the main text, but using the full 
avoidance resistance and tolerance models. The key goal of this script to to demonstrate
that the inference we are making from the reduced, moment-closure models is
the same as the full model. Here, we approximate the full models with 200 bins.
"""


def make_partial_correlation_plots(results, param_value, letters):
	""" Make the partial correltaion plots from the simulation """

	(magnitude_of_decline, time_decline_stop, time_full_recovery,
	 inds_used, parasiter_vals, parasiteR0_vals, hostr_vals,
	 initial_cv, check_convergence) = results

	output = [time_decline_stop, time_full_recovery, magnitude_of_decline]
	ylabels = ["Time until recovery starts", "Time until recovery stops", "Magnitude of decline"]

	# Use these conditions to eliminate boundary conditions and simulation 
	# problems
	cond = (pd.Series(time_decline_stop <= 0.3) |
	        pd.Series(parasiter_vals <= 10**-0.5) |
	        pd.Series(parasiteR0_vals > 20) |
	        pd.Series(magnitude_of_decline < percent) |
	        pd.Series(time_full_recovery).isna() |
	        pd.Series(check_convergence == False)).values
	which_inds = np.where(cond)[0]
	good_inds = np.where(~cond)[0] # Inds that have some decline

	fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
	axes = axes.ravel()

	for i, out in enumerate(output):
	    
	    dat = pd.DataFrame({'x1': np.log(parasiter_vals[good_inds]), 
	                        'x2': np.log(hostr_vals[good_inds]),
	                        'x3': np.log(initial_cv[good_inds]),
	                        'y': np.log(out[good_inds])})
	    xnames = {'x1': "Parasite\ngrowth\nrate", 
	              'x2': "Host\ngrowth\nrate",
	              'x3': "Initial CV\n" + r"in $\{0}$".format(param_value)}
	    xnm = list(xnames.keys())
	    
	    y, fullX = dmatrices("y ~ scale(x1) + scale(x2) + scale(x3)", data=dat, return_type="dataframe")
	    fullfit = GLM(y, fullX).fit()
	    fullresid = fullfit.resid_pearson
	    R2 = 1 - (np.sum(fullresid**2) / np.sum((y.values - np.mean(y.values))**2))
	    coefs = fullfit.params[1:]
	    
	    pcorr_all = {}
	    for x in xnm:
	        newx = list(xnames.keys())
	        newx.remove(x)
	        pcorr = partial_corr(data=dat, x=x, y="y", covar=newx, method="pearson")
	        pcorr_all[xnames[x]] = [pcorr.r.values[0]]
	    
	    sns.barplot(pd.DataFrame(pcorr_all), ax=axes[i], palette=np.repeat(sns.color_palette().as_hex()[i], 3))
	    xlim = axes[i].get_xlim()
	    axes[i].hlines(0, *xlim, color='black', linewidth=0.5)
	    axes[i].set_xlim(*xlim)
	    axes[i].spines['right'].set_visible(False)
	    axes[i].spines['top'].set_visible(False)
	    #axes[i].set_title(ylabels[i])
	    axes[i].text(-0.05, 1.05, letters[i], size=12, ha='center', transform=axes[i].transAxes)
	    axes[i].text(0.5, 0.9, "$R^2$ = {0:.2}".format(R2), ha='center', transform=axes[i].transAxes)
	    axes[i].set_ylim(-1.01, 1.01)
	    xticks = axes[i].get_xticks()
	    for t, cf in enumerate(coefs):
	        axes[i].text(xticks[t], 0, "{0:.2}".format(cf), ha='center')

	axes[0].set_ylabel("Partial Correlation Coefficient")
	fig.savefig("../results/{0}_full_model_pcc.pdf".format(model_type), bbox_inches="tight")
	return((fig, axes))

if __name__ == '__main__':


	samps = 1000 # Number of parameters to explore
	percent = 0.02 # Recovery threshold
	plist = ['beta', 'lam', 'mu_z', 'r', 'mu', 'init_k', 'alpha', 'delta']
	lower = [0.05, 100,  10,    0.1, 0.1, 0.5,  1, 0.1]
	upper = [0.3,  1000, 100,   10,  3,    5,   3, 0.5]
	sampler = qmc.LatinHypercube(d=len(lower))
	sample = sampler.random(samps)
	lhc_params = qmc.scale(sample, lower, upper)
	
	# Set the model type: either resistance ('beta') or tolerance ('alpha')
	model_types = ['alpha', 'beta']
	param_range = {'alpha': (0, 15), 'beta': (0, 2)}
	model_function = {'alpha': eee.full_tolerance_model, 
										'beta': eee.full_resistance_model}
	labels = {'alpha': "tolerance",
	 		  'beta': 'resistance'}

	letters = {'alpha': ['A,', "B.", "C."],
			   'beta': ['D.', "E.", "F."]}

	bins = 200 # Number of classes with which to approximate the infinite ODE

	for model_type in model_types:

		# Predictors
		parasiter_vals = []
		parasiteR0_vals = []
		hostr_vals = []
		initial_cv = []
		delta_vals = []

		# Output
		time_decline_stop = []
		time_full_recovery = []
		magnitude_of_decline = []
		inds_used = []
		trajectories = []
		params_used = []
		check_convergence = []

		for s in range(samps):

			print("Model {0}, simulation {1}".format(model_type, s))
			params = {}

			# Ensure no trade-off
			params['beta_m'] = 1e-10
			params['alpha_m'] = 1e-10

			# Assign parameters
			for i, p in enumerate(plist):
				params[p] = lhc_params[s, i]
			params['bins'] = bins

			# Set up the initial condition Gamma distribution
			init_var = (params[model_type]**2 / params['init_k']) + params[model_type]**2
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
			params['r_vals'] = eee.r_fxn(p_vals_mid, 0, params['r'])

			# Set-up the full model simulation
			N_init = (params['r'] - params['mu']) / params['delta']
			Nvals = probs_norm * N_init
			Ivals = np.zeros(len(Nvals))
			init_vals = np.concatenate([Nvals, Ivals, [1e-5]])
			parasiteR0 = ((N_init*params['beta']*params['lam']) / 
									 ((params['alpha'] + params['mu'])*params['mu_z']))
			hostr = params['r'] - params['mu']

			if parasiteR0 > 1 and hostr > 0:

				parasiter = eee.max_eig(N_init, params)
				    
				if parasiter > 0.01:

					time = np.linspace(0, 500, num=1000)

					res, out = odeint(model_function[model_type], init_vals, time, 
													  (params, ), full_output=True)
					res = pd.DataFrame(res).assign(time=time)
					check_convergence.append(out['message'] == "Integration successful.")
					full_N = res.iloc[:, :params['bins']].sum(axis=1)

					# Get recovery metrics
					recov = eee.recovery_metrics(full_N, time, percent=percent, change_point=False)

					# Save results
					magnitude_of_decline.append(recov[0])
					time_decline_stop.append(recov[1])
					time_full_recovery.append(recov[2])
					inds_used.append(s)
					parasiter_vals.append(parasiter)
					parasiteR0_vals.append(parasiteR0)
					hostr_vals.append(hostr)
					delta_vals.append(params['delta'])
					initial_cv.append(np.sqrt(init_var) / params[model_type])
					trajectories.append(res)
					params_used.append(params)

		# convert to arrays
		magnitude_of_decline = np.array(magnitude_of_decline)
		time_decline_stop = np.array(time_decline_stop)
		time_full_recovery = np.array(time_full_recovery)
		inds_used = np.array(inds_used)
		parasiter_vals = np.array(parasiter_vals)
		parasiteR0_vals = np.array(parasiteR0_vals)
		hostr_vals = np.array(hostr_vals)
		initial_cv = np.array(initial_cv)
		check_convergence = np.array(check_convergence)

		all_results = (magnitude_of_decline, time_decline_stop, time_full_recovery,
									 inds_used, parasiter_vals, parasiteR0_vals, hostr_vals,
									 initial_cv, check_convergence)

		# Plot results
		fig, axes = make_partial_correlation_plots(all_results, model_type, letters[model_type])
