# ALS.py
# Author: Matthew Smarte
# Version: 1.2.0
# Last Updated: 01/12/19

# This code is designed to be imported and run inside a Jupyter notebook using an iPython kernel.

'''
ALS.py DEPENDENCIES

This code was developed and tested using the following packages/versions.
Other versions may also work.

python 		3.6.1
numpy 		1.12.1
pandas 		0.20.1
scipy 		0.19.0
matplotlib 	2.2.2
ipython 	5.3.0

ALS.py MAJOR UPDATE HISTORY

1.0.0 - 07/10/18 - Initial release (__init__, fit, plot_data_model, plot_model, _time_axis, _model, and _conv_IRF methods).
1.1.0 - 07/25/18 - Added bootstrap method.
1.1.1 - 10/04/18 - Added conc_units kwarg for fit, plot_data_model, and boostrap methods.
1.1.2 - 01/09/19 - Added quiet kwarg to bootstrap method.
1.2.0 - 01/12/19 - Added preliminary monte_carlo_params method.
'''

# TODO:
# Add checks to make sure the inputs from the user have the correct formatting, throw errors if they are not (ex: df_data err field can have no zeros)
# Adjust random sampling methodology of monte carlo parameter sampling to accomodate parameters with negative values and correlation between parameters 

import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, clear_output

class KineticModel:

	_dt = 0.02	# Fundamental kinetic time step (ms)

	def __init__(self, user_model, err_weight=True, fit_pre_photo=False, apply_IRF=True, apply_PG=True, t_PG=1.0):
		'''
		Initializor for a KineticModel object.
		See ex_notebook_1.ipynb for API documentation.
		'''

		self._user_model = user_model
		self._err_weight = err_weight
		self._fit_pre_photo = fit_pre_photo
		self._apply_IRF = apply_IRF
		self._apply_PG = apply_PG
		self._t_PG = t_PG

	def fit(self, t, tbin, df_data, df_model_params, df_ALS_params, delta_xtick=20.0, conc_units=False, save_fn=None, quiet=False, **kwargs):
		'''
		Method for fitting data and optimizing parameters.
		See ex_notebook_1.ipynb for API documentation.
		'''
		
		# Check fit t0 / fit_pre_photo
		if df_ALS_params.at['t0','fit'] and not self._fit_pre_photo:
			print('ERROR: Fit t0 is True but fit_pre_photo is False!')
			print('If we are fitting t0, then we must also fit the pre-photolysis data.')
			print('Otherwise, the cost function would be minimized by arbitrarily increasing t0.')
			return None, None, None, None, None, None

		# Determine start time, end time, and range of times over which to fit
		t_start = float(t.min())
		t_end = float(t.max())
		idx_cost = np.full(t.shape, True) if self._fit_pre_photo else (t >= df_ALS_params.at['t0','val']) # Boolean array of indices to use in cost calculation

		# Establish corrspondence between data and model time axes
		# Each entry in t has an exact match to an entry in t_model
		# We take the below approach (rather than ==) to prevent any problems with numerical roundoff
		t_model = self._time_axis(t_start, t_end, tbin)
		idx_model = [np.abs(t_model - t[_]).argmin() for _ in range(t.size)] # Array of positional indices, maps model axis --> data axis

		# Organize fitted species data into data_val and data_err frames
		# Columns of data_val and data_err are species and the rows correspond to times in t array
		data_fit = df_data[df_data['fit']]
		species_names = list(data_fit.index)
		data_val = pd.DataFrame(list(data_fit['val']), index=species_names).T
		if self._err_weight:
			data_err = pd.DataFrame(list(data_fit['err']), index=species_names).T

		# Organize the fit parameters
		model_params_fit = df_model_params[df_model_params['fit']] 
		ALS_params_fit = df_ALS_params[df_ALS_params['fit']]
		p_names = list(model_params_fit.index) + list(ALS_params_fit.index)
		p0 = np.concatenate((model_params_fit['val'], ALS_params_fit['val']))

		# Define the cost function to be optimized
		def calc_cost(p):

			# Organize parameter values used for the current iteration of the fit into dictionaries
			model_params_p = {}
			for param in df_model_params.index:
				model_params_p[param] = p[p_names.index(param)] if param in p_names else df_model_params.at[param,'val']
			ALS_params_p = {}
			for param in df_ALS_params.index:
				ALS_params_p[param] = p[p_names.index(param)] if param in p_names else df_ALS_params.at[param,'val']

			# Run the model - we only need the concentrations dataframe
			_, c_model = self._model(t_start, t_end, tbin, model_params_p, ALS_params_p)

			# Calculate the weighted residual array across points included in the cost computation
			res = []
			for species in species_names:
				obs = data_val[species]
				mod = ALS_params_p['S_'+species]*c_model[species]

				# We take the sqrt of the weight since leastsq will square the array later
				# Important to perform .values conversion to array BEFORE we subtract obs and mod (pandas subtracts Series by index agreement not position)
				species_res = np.sqrt(data_fit.at[species,'weight']) * (obs.values - mod.values[idx_model])[idx_cost]

				if self._err_weight:
					err = data_err[species]
					species_res = species_res / err.values[idx_cost]

				res.append(species_res)

			# leastsq will square then sum all entries in the returned array, and minimize this cost value
			return np.concatenate(res)

		if not quiet:
			print('Initial Cost Function Value: {:g}'.format((calc_cost(p0)**2).sum()))
			print()

		# Perform the fit
		# NOTE: The backend of leastsq will automatically autoscale the fit parameters to the same order of magnitude if diag=None (default).
		p, cov_p, infodict, mesg, ier = leastsq(calc_cost, p0, full_output=True, **kwargs)
		# Calculate minimized cost value
		cost = (infodict['fvec']**2).sum()

		# Prepare covariance and correlation matrices
		if cov_p is not None:

			# Scale the covariance matrix according to documentation for leastsq and source code for curve_fit.
			# Scale factor is cost / (# of data points - # fit parameters).
			M = infodict['fvec'].size
			N = p.size
			cov_p *= cost / (M-N)

			# Compute standard errors
			p_err = np.sqrt(np.diag(cov_p))

			# Compute correlation matrix
			corr_p = np.array([[cov_p[i][j] / (np.sqrt(cov_p[i][i]*cov_p[j][j])) for j in range(cov_p.shape[1])] for i in range(cov_p.shape[0])])

		else:
			p_err = np.full(p.shape, np.NaN)
			corr_p = None

		# Convert fit results to dataframes
		df_p = pd.DataFrame(np.array((p,p_err)).T, index=p_names, columns=('val','err'))
		df_cov_p = pd.DataFrame(cov_p, index=p_names, columns=p_names)
		df_corr_p = pd.DataFrame(corr_p, index=p_names, columns=p_names)

		if not quiet:
			# Display results
			print('Optimization terminated successfully.' if ier in (1,2,3,4) else 'Optimization FAILED.')
			print('Exit Code = {:d}'.format(ier))
			print('Exit Message = {}'.format(mesg))
			print()

			print('Optimized Cost Function Value = {:g}'.format(cost))
			print()

			print('Optimized Parameters and Estimated Standard Errors:')
			display(df_p)
			print()

			print('Estimated Correlation Matrix:')
			display(df_corr_p)
			print()

			# Plot the fits
			df_model_params_p = df_model_params.copy()
			df_ALS_params_p = df_ALS_params.copy()

			for param in df_p.index:
				if param in df_model_params_p.index:
					df_model_params_p.at[param,'val'] = df_p.at[param,'val']
				else:
					df_ALS_params_p.at[param,'val'] = df_p.at[param,'val']

			self.plot_data_model(t, tbin, df_data, df_model_params_p, df_ALS_params_p, delta_xtick=delta_xtick, conc_units=conc_units, save_fn=save_fn, print_cost=False)

		return df_p, df_cov_p, df_corr_p, cost, mesg, ier

	def plot_data_model(self, t, tbin, df_data, df_model_params, df_ALS_params, delta_xtick=20.0, conc_units=False, save_fn=None, print_cost=True):
		'''
		Method for plotting the scaled model overlaid on the inputted species data. (No fit is performed.)
		Residuals are also plotted and the value of the cost function with these parameters is optionally outputted.
		Only species for which fit is True in df_data are shown and included in the cost calculation.
		See ex_notebook_1.ipynb for API documentation.
		'''

		# Run the model
		t_start = float(t.min())
		t_end = float(t.max())
		t_model, c_model = self._model(t_start, t_end, tbin, df_model_params['val'].to_dict(), df_ALS_params['val'].to_dict())

		# Only plot the species for which fit=True
		# Columns of data_val and data_err are species and the rows correspond to times in t array
		data_fit = df_data[df_data['fit']]
		species_names = list(data_fit.index)
		nSpecies = len(species_names)
		data_val = pd.DataFrame(list(data_fit['val']), index=species_names).T
		if self._err_weight:
			data_err = pd.DataFrame(list(data_fit['err']), index=species_names).T

		# Setup for residual and cost computations
		cost = 0
		idx_cost = np.full(t.shape, True) if self._fit_pre_photo else (t >= df_ALS_params.at['t0','val']) # Boolean array of indices to use in cost calculation
		idx_model = [np.abs(t_model - t[_]).argmin() for _ in range(t.size)] # Array of positional indices, maps model axis --> data axis

		# Set up the grid of subplots
		ncols = 3
		nrows = (nSpecies//ncols) if (nSpecies%ncols) == 0 else (nSpecies//ncols)+1
		dpi = 120

		plt.rc('font', size=9)
		plt.rc('axes.formatter', useoffset=False)
		f = plt.figure(figsize=(1000/dpi,450*nrows/dpi), dpi=dpi)
		gs = gridspec.GridSpec(nrows, ncols, figure=f, hspace=0.3, wspace=0.3, top=0.9)

		# Determine x-axis ticks
		tick_low = (t_start//delta_xtick)*delta_xtick
		tick_high = t_end if t_end % delta_xtick == 0. else ((t_end//delta_xtick)+1)*delta_xtick
		ticks = np.linspace(tick_low, tick_high, num=round(((tick_high-tick_low)/delta_xtick)+1), endpoint=True)

		# Make the subplots
		c_data = []
		s_model = []
		for i, species in enumerate(species_names):
			obs = data_val[species]
			mod = df_ALS_params.at['S_'+species,'val']*c_model[species]
			s_model.append(mod)

			c_obs = data_val[species]/df_ALS_params.at['S_'+species,'val']
			c_mod = c_model[species]
			c_data.append(c_obs)
			
			# Compute this species' residual and cost contribution
			# Important to perform .values conversion to array BEFORE we subtract obs and mod (pandas subtracts Series by index agreement not position)
			res = obs.values - mod.values[idx_model]
			cost_i = np.sqrt(data_fit.at[species,'weight']) * res[idx_cost]
			if self._err_weight:
				err = data_err[species]
				cost_i = cost_i / err.values[idx_cost]
			cost += (cost_i**2).sum()

			c_res = c_obs.values - c_mod.values[idx_model]

			j = i // 3	# Row index
			k = i % 3	# Col index

			gs_jk = gridspec.GridSpecFromSubplotSpec(2, 1, hspace=0, height_ratios=[3,1], subplot_spec=gs[j,k])
			ax0 = plt.subplot(gs_jk[0])	# Data & Model
			ax1 = plt.subplot(gs_jk[1])	# Data - Model

			if conc_units:
				ax0.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
				ax0.plot(t, c_obs, 'o')					# Plot the data
				ax0.plot(t_model, c_mod, linewidth=2)	# Plot the model

				# OOM calculation adapted from matplotlib.ticker.ScalarFormatter._set_orderOfMagnitude source code
				oom = np.floor(np.log10(ax0.get_yticks().max()))

				ax1.ticklabel_format(axis='y', style='plain')
				ax1.plot(t, c_res/(10**oom), 'o')		# Plot residual
			else:
				ax0.plot(t, obs, 'o')					# Plot the data
				ax0.plot(t_model, mod, linewidth=2)		# Plot the model
				ax1.plot(t, res, 'o')					# Plot residual

			ax1.plot(t_model, np.zeros(t_model.shape))	# Plot zero residual line

			# Manually set x-axis ticks
			ax0.set_xticks(ticks)
			ax1.set_xticks(ticks)

			# Labels
			ax0.set_title(species, fontweight='bold')	 # Make the title the species name
			ax0.set_xticklabels([])						 # Hide x-axis tick labels for top plot
			ax1.set_xlabel('Time (ms)')					 # Set x-axis label for bottom plot
			if k == 0:									 # Set y-axis labels if plot is in first column
				ax0.set_ylabel('Data & Model')
				ax1.set_ylabel('Data - Model')

		plt.show()

		# Print the value of the cost function
		if print_cost:
			print()
			print('Cost Function Value = {:g}'.format(cost))

		# Save the scaled model traces
		if save_fn:
			append_data = (lambda _ : _ + '-data')
			append_model = (lambda _ : _ + '-model')

			if conc_units:
				df_data_out = pd.DataFrame(c_data).T.rename(columns=append_data)
				df_model_out = c_model.rename(columns=append_model) 
			else:
				df_data_out = data_val.rename(columns=append_data)
				df_model_out = pd.DataFrame(s_model).T.rename(columns=append_model)

			df_data_out.insert(0,'t-data',t)
			df_model_out.insert(0,'t-model',t_model)

			df = pd.concat((df_data_out,df_model_out), axis='columns')
			df.to_csv(save_fn, index=False)

	def plot_model(self, t_start, t_end, tbin, df_model_params, df_ALS_params, delta_xtick=20.0, save_fn=None):
		'''
		Plots the model in concentration units (molc/cm3) without the data.
		All species returned by the user model function are shown.
		See ex_notebook_1.ipynb for API documentation.
		'''

		# Run the model
		t_model, c_model = self._model(t_start, t_end, tbin, df_model_params['val'].to_dict(), df_ALS_params['val'].to_dict())
		species_names = list(c_model.columns)
		nSpecies = len(species_names)

		# Set up the grid of subplots
		ncols = 3
		nrows = (nSpecies//ncols) if (nSpecies%ncols) == 0 else (nSpecies//ncols)+1
		dpi = 120

		plt.rc('font', size=9)
		plt.rc('axes.formatter', useoffset=False)
		f = plt.figure(figsize=(1000/dpi,325*nrows/dpi), dpi=dpi)
		gs = gridspec.GridSpec(nrows, ncols, figure=f, hspace=0.45, wspace=0.3, top=0.9, bottom=0.2)

		# Determine x-axis ticks
		tick_low = (t_start//delta_xtick)*delta_xtick
		tick_high = t_end if t_end % delta_xtick == 0. else ((t_end//delta_xtick)+1)*delta_xtick
		ticks = np.linspace(tick_low, tick_high, num=round(((tick_high-tick_low)/delta_xtick)+1), endpoint=True)

		# Make the subplots
		s_model = []
		for i, species in enumerate(species_names):
			mod = c_model[species]
			s_model.append(mod)

			j = i // 3	# Row index
			k = i % 3	# Col index

			ax = plt.subplot(gs[j,k])
			ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
			ax.plot(t_model, mod, linewidth=2)

			# Manually set x-axis ticks
			ax.set_xticks(ticks)

			# Labels
			ax.set_title(species, fontweight='bold')	 # Make the title the species name
			ax.set_xlabel('Time (ms)')					 # Set x-axis label for bottom plot
			if k == 0:									 # Set y-axis labels if plot is in first column
				ax.set_ylabel('Concentration ($\mathregular{molc/cm^{3}})$')

		plt.show()

		# Save the model traces
		if save_fn:
			df = pd.DataFrame(s_model).T
			df.insert(0,'t',t_model)
			df.to_csv(save_fn, index=False)
	
	def bootstrap(self, t, tbin, df_data, df_model_params, df_ALS_params, N, delta_xtick=20.0, conc_units=False, save_fn=None, quiet=False, **kwargs):
		'''
		Performs a bootstrap simulation to estimate the covariance matrix of the fit parameters.
		See ex_notebook_1.ipynb for API documentation.
		'''

		# Check fit t0 / fit_pre_photo
		if df_ALS_params.at['t0','fit'] and not self._fit_pre_photo:
			print('ERROR: Fit t0 is True but fit_pre_photo is False!')
			print('If we are fitting t0, then we must also fit the pre-photolysis data.')
			print('Otherwise, the cost function would be minimized by arbitrarily increasing t0.')
			return None, None, None, None

		# Determine range of data over which to generate bootstrap samples
		idx_cost = np.full(t.shape, True) if self._fit_pre_photo else (t >= df_ALS_params.at['t0','val'])
		M = sum(idx_cost)	# Length of each bootstrap sample

		dist_p = []
		N_success = 0
		N_fail = 0

		while N_success < N:
			if not quiet:
				print('Successful Iterations: {:d}'.format(N_success))
				print('Failed Iterations: {:d}'.format(N_fail))
				print()
				print('Current Iteration: {:d}'.format(N_success+N_fail+1))
				clear_output(wait=True)

			# Randomly generate indices (with replacement)
			idx_b = np.random.choice(M, M)

			# Create the bootstrap sample
			t_b = t[idx_cost][idx_b]
			df_data_b = df_data.copy()
			for species in df_data_b.index:
				df_data_b.at[species,'val'] = df_data_b.at[species,'val'][idx_cost][idx_b]
				df_data_b.at[species,'err'] = df_data_b.at[species,'err'][idx_cost][idx_b]

			# Fit the bootstrap sample
			df_p_b, _, _, cost_b, mesg_b, ier_b = self.fit(t_b, tbin, df_data_b, df_model_params, df_ALS_params, quiet=True, **kwargs)
			success = (ier_b in (1,2,3,4))

			# Save the output as the function runs so it may be accessed during a simulation
			# Note: We use pd.DataFrame.to_csv because pd.Series.to_csv orients as column instead of row
			if save_fn:
				mesg_b_adj = mesg_b.replace('\n','').replace(',','') # Remove newlines and commas
				row_save = pd.DataFrame(df_p_b['val'].append(pd.Series((cost_b,success,ier_b,mesg_b_adj), index=('cost','success','ier','mesg')))).T
				if (N_success+N_fail) == 0:
					# First iteration - create a new file containing first row and column headers
					row_save.to_csv(save_fn, index=False)
				else:
					# Subsequent iterations - append row to file without headers
					row_save.to_csv(save_fn, index=False, header=False, mode='a')

			# If fit was successful, add results to df_dist_p for calculating statistics after loop is finished
			if success:
				dist_p.append(df_p_b['val'])
				N_success +=1
			else:
				N_fail += 1

		df_dist_p = pd.DataFrame(dist_p, index=pd.RangeIndex(len(dist_p)))

		# Summary statistics
		# The estimated standard error is the standard deviation of the bootstrap distribution 
		df_p = pd.DataFrame([df_dist_p.mean(),df_dist_p.std()], index=('val','err')).T
		df_cov_p = df_dist_p.cov()
		df_corr_p = df_dist_p.corr()

		if not quiet:
			# Display results
			print('Bootstrap simulation completed.')
			print('Successful Iterations: {:d}'.format(N_success))
			print('Failed Iterations: {:d}'.format(N_fail))
			print()
			print('Returned variables and below summary include only successful iterations.')
			if save_fn:
				print('Results saved to file include all iterations.')
			print()

			print('Average Parameter Values and Estimated Standard Errors:')
			display(df_p)
			print()

			print('Estimated Correlation Matrix:')
			display(df_corr_p)
			print()

			print('Below plots and cost use the average parameter values:')

			# Make plots
			df_model_params_p = df_model_params.copy()
			df_ALS_params_p = df_ALS_params.copy()

			for param in df_p.index:
				if param in df_model_params_p.index:
					df_model_params_p.at[param,'val'] = df_p.at[param,'val']
				else:
					df_ALS_params_p.at[param,'val'] = df_p.at[param,'val']

			self.plot_data_model(t, tbin, df_data, df_model_params_p, df_ALS_params_p, delta_xtick=delta_xtick, conc_units=conc_units, save_fn=None, print_cost=True)

		return df_p, df_cov_p, df_corr_p, df_dist_p

	def monte_carlo_params(self, t, tbin, df_data, df_model_params, df_ALS_params, N, delta_xtick=20.0, conc_units=False, save_fn=None, quiet=False, **kwargs):
		'''
		Performs a monte carlo simulation over fixed parameters to assess how their uncertainties propagate into uncertainties of the fitted parameters.
		See ex_notebook_1.ipynb for API documentation.

		The random sampling of the fitted parameters is preliminary (see below).  Future updates to this method should:
		--- Allow fixed parameters for which negative values are physically meaningful to be included
		--- Take into acccount possible correlation between fixed parameters
		See full discussion of these issues in ex_notebook_1.ipynb. 
		'''

		# Check fit t0 / fit_pre_photo
		if df_ALS_params.at['t0','fit'] and not self._fit_pre_photo:
			print('ERROR: Fit t0 is True but fit_pre_photo is False!')
			print('If we are fitting t0, then we must also fit the pre-photolysis data.')
			print('Otherwise, the cost function would be minimized by arbitrarily increasing t0.')
			return None, None, None, None

		# Identify parameters to vary - fit field must be False and err field must be nonzero
		model_params_vary = df_model_params[(df_model_params['fit'] == False) & (df_model_params['err'] != 0)] 
		ALS_params_vary = df_ALS_params[(df_ALS_params['fit'] == False) & (df_ALS_params['err'] != 0)]

		# Parameter sampling is currently done using the truncated normal distribution:
		# --- Lower bound of distribution (lb) is 0.
		# --- Upper bound of distribution (ub) is 2 times the parameter value.
		# --- Mean of distribution (mu) is the parameter value ('val' field).
		# --- Standard deviation of distribution (sigma) is the parameter uncertainty ('err' field).
		# See ex_notebook_1.ipynb for a discussion on the benefits / limitations of this approach.
		# To sample from scipy.stats.truncnorm, use truncnorm.rvs((lb-mu)/sigma, (ub-mu)/sigma, mu, sigma).   
		# For our choice of lb and ub, this reduces to the formula used below.
		def sample(mu, sigma):
			return truncnorm.rvs(-mu/sigma, mu/sigma, mu, sigma)

		dist_p = []
		N_success = 0
		N_fail = 0

		while N_success < N:
			if not quiet:
				print('Successful Iterations: {:d}'.format(N_success))
				print('Failed Iterations: {:d}'.format(N_fail))
				print()
				print('Current Iteration: {:d}'.format(N_success+N_fail+1))
				clear_output(wait=True)

			# Create the monte carlo parameter sample
			p_vary_vals = {}

			df_model_params_mc = df_model_params.copy()
			for p_row in model_params_vary.itertuples():
				df_model_params_mc.at[p_row.Index,'val'] = p_vary_vals[p_row.Index] = sample(p_row.val, p_row.err)
				
			df_ALS_params_mc = df_ALS_params.copy()
			for p_row in ALS_params_vary.itertuples():
				df_ALS_params_mc.at[p_row.Index,'val'] = p_vary_vals[p_row.Index] = sample(p_row.val, p_row.err)
			
			# Perform fit using the parameter sample
			df_p_mc, _, _, cost_mc, mesg_mc, ier_mc = self.fit(t, tbin, df_data, df_model_params_mc, df_ALS_params_mc, quiet=True, **kwargs)
			success = (ier_mc in (1,2,3,4))

			# Save the output as the function runs so it may be accessed during a simulation
			# Note: We use pd.DataFrame.to_csv because pd.Series.to_csv orients as column instead of row
			if save_fn:
				mesg_mc_adj = mesg_mc.replace('\n','').replace(',','') # Remove newlines and commas

				row_save = pd.Series(p_vary_vals) # Create Series with values of the randomly sampled parameters
				row_save = row_save.append(pd.Series((cost_mc,success,ier_mc,mesg_mc_adj), index=('cost','success','ier','mesg'))) # Append fit result info
				row_save = row_save.append(df_p_mc['val']) # Append fitted parameter values
				row_save = pd.DataFrame(row_save).T # Convert to DataFrame for saving

				if (N_success+N_fail) == 0:
					# First iteration - create a new file containing first row and column headers
					row_save.to_csv(save_fn, index=False)
				else:
					# Subsequent iterations - append row to file without headers
					row_save.to_csv(save_fn, index=False, header=False, mode='a')

			# If fit was successful, add results to df_dist_p for calculating statistics after loop is finished
			if success:
				dist_p.append(df_p_mc['val'])
				N_success +=1
			else:
				N_fail += 1

		df_dist_p = pd.DataFrame(dist_p, index=pd.RangeIndex(len(dist_p)))

		# Summary statistics
		# The estimated standard error is the standard deviation of the monte carlo distribution
		df_p = pd.DataFrame([df_dist_p.mean(),df_dist_p.std()], index=('val','err')).T
		df_cov_p = df_dist_p.cov()
		df_corr_p = df_dist_p.corr()

		if not quiet:
			# Display results
			print('Monte carlo simulation completed.')
			print('Successful Iterations: {:d}'.format(N_success))
			print('Failed Iterations: {:d}'.format(N_fail))
			print()
			print('Returned variables and below summary include only successful iterations.')
			if save_fn:
				print('Results saved to file include all iterations.')
			print()

			print('Average Fitted Parameter Values and Estimated Standard Errors:')
			display(df_p)
			print()

			print('Estimated Correlation Matrix:')
			display(df_corr_p)
			print()

			print('Below plots and cost use the average parameter values:')

			# Make plots
			df_model_params_p = df_model_params.copy()
			df_ALS_params_p = df_ALS_params.copy()

			for param in df_p.index:
				if param in df_model_params_p.index:
					df_model_params_p.at[param,'val'] = df_p.at[param,'val']
				else:
					df_ALS_params_p.at[param,'val'] = df_p.at[param,'val']

			self.plot_data_model(t, tbin, df_data, df_model_params_p, df_ALS_params_p, delta_xtick=delta_xtick, conc_units=conc_units, save_fn=None, print_cost=True)

		return df_p, df_cov_p, df_corr_p, df_dist_p

	def _time_axis(self, t_start, t_end, tbin):
		'''
		Private method for computing time axes.

		Parameters:
		t_start = float: start time of time axis (inclusive), must be integer multiple of tbin*dt
		t_end = float: end time of time axis (inclusive), must be integer multiple of tbin*dt
		tbin = int: the axis will be evenly spaced by tbin*dt

		Returns:
		t = ndarray: the time axis
		'''
		return np.linspace(t_start, t_end, num=round(((t_end-t_start)/(tbin*self._dt))+1), endpoint=True)

	def _model(self, t_start, t_end, tbin, model_params, ALS_params):
		'''
		Private method for integrating the user model.
		Implements the photolysis gradient, instrument response function, and photolysis offset.
		A copy is created for anything passed to the user model to prevent problems with any mutable objects.
	
		Procedure:
		1)	A time axis is created for running the model, with photolysis occurring at 0 ms.  
			Forced to have at least 20 ms of pre-photolysis baseline to ensure correct IRF convolution of points immediately after photolysis.
		2)	User model is integrated in steps (set by t_PG) with different initial radical concentrations (set by X0 and B).
			For example, if t_PG = 1 ms, then:
				- The 0-1 ms output is determined by integrating over 0-1 ms with initial radical concentration X0*(1+B*(0.5 ms)).
				- The 1-2 ms output is determined by integrating over 0-2 ms with initial radical concentration X0*(1+B*(1.5 ms)), and taking the 1-2 ms portion.
				- The 2-3 ms output is determined by integrating over 0-3 ms with initial radical concentration X0*(1+B*(2.5 ms)), and taking the 2-3 ms portion.
				- Etc.
		3)	The concentration profiles are convolved with the IRF (see conv_IRF).
		4)	The time axis is adjusted for the photolysis offset t0.
		5) 	The profiles are trimmed and averaged within time bins to achieve a step size of dt*tbin over [t_start, t_end].

		Parameters:
		t_start = float
			start time of the model output (ms), must be integer multiple of tbin*dt, must be greater than or equal to -20 ms
		t_end = float
			end time of the model output (ms), must be integer multiple of tbin*dt, must be greater than t_start
		tbin = int
			step size of the model output will be tbin*dt
		model_params = dict
			keys (str) are model param names and values (float) are the parameter values
			X0 is required
		ALS_params = dict
			keys (str) are ALS param names and values (float) are the parameter values
			t0 is required, A is required if apply_IRF = True, B is required if apply_PG = True

		Returns:
		t_model = ndarray
			Time axis of the model output (ms).  Evenly spaced by tbin*dt over [t_start, t_end].
		c_model = DataFrame
			Columns correspond to the species returned by the user model.
			Rows correspond to the modeled concentrations at the times in t_model
		'''

		# Create time axis for running the model (before applying tbin and t0)
		t0 = float(ALS_params['t0'])
		if t0 == 0:
			t = self._time_axis(-20, t_end+((tbin-1)*self._dt), 1)
		elif t0 < 0:
			t = self._time_axis(-20, t_end+((tbin-1)*self._dt)-t0, 1)
		elif t0 > 0:
			t = self._time_axis(-20-t0, t_end+((tbin-1)*self._dt), 1)
		
		# Model integration w/ photolysis gradient
		if self._apply_PG:

			X0 = model_params['X0']
			B = ALS_params['B']
			idx_zero = np.abs(t).argmin()

			# First populate concentration matrix for t < 0
			m, c = self._user_model(t[:idx_zero].copy(), model_params.copy())

			# Then populate concentration matrix for t >= 0
			idx_curr = idx_zero
			idx_step = round(self._t_PG/self._dt)

			while idx_curr < t.size:
				t_mid = t[idx_curr] + self._t_PG/2
				X_curr = X0*(1+B*t_mid)

				model_params_curr = model_params.copy()
				model_params_curr['X0'] = X_curr

				# If idx_curr+idx_step >= t.size, then t[:idx_curr+idx_step] is equivalent to t[:]
				c_tmp = self._user_model(t[:idx_curr+idx_step].copy(), model_params_curr)[1]

				for species in c:
					c[species] = np.append(c[species], c_tmp[species][idx_curr:])

				idx_curr += idx_step	

		# Model integration w/o photolysis gradient
		else:
			m, c = self._user_model(t.copy(), model_params.copy())

		# Convolve with IRF
		if self._apply_IRF:
			c = self._conv_IRF(m, c, t.size, ALS_params['A'])

		# Apply photolysis offset (the sign of this is correct)
		t = t + t0

		# Trim data to desired range
		idx_start = (np.abs(t - t_start)).argmin()
		idx_end   = (np.abs(t - (t_end+((tbin-1)*self._dt)))).argmin()

		t = t[idx_start:idx_end+1]
		for species in c:
			c[species] = c[species][idx_start:idx_end+1]

		# Bin the model output
		t_out = self._time_axis(t_start, t_end, tbin)
		c_out = {}
		for species in c:
			c_out[species] = np.zeros(t_out.size)

		# This is the same method for binning used in the IGOR code (although they sum instead of average)
		for i in range(t_out.size):
			for species in c_out:
				c_out[species][i] = np.mean(c[species][i*tbin:(i+1)*tbin])

		# Columns of dataframe are species, rows correspond to times in t_out
		return t_out, pd.DataFrame.from_dict(c_out)

	def _conv_IRF(self, m, y, N, A):
		'''
		Private method for convolving signals with the mass-dependent ALS instrument response function.

		At the ALS, molecules exit the pinhole with a Maxwell-Boltzmann distribution of velocities.
		The ion signal at a specific observation time therefore arises from molecules that exit the
		pinhole at a distribution of kinetic times that all come before the observation time.  See
		more details here: https://doi.org/10.1002/kin.20262

		Each convolved point is thus calculated as a weighted average of all points in the model at 
		earlier times, where the weighting is given by the mass-dependent IRF function.

		Note that this operation is not quite a "convolution" as one might usually think about it,
		since a convolved point only contains contributions from the modeled points at earlier times,
		and has zero contribution from modeled points at future times.

		Parameters:
		m = dictionary: keys are species (str), values are their masses (float - amu)
		y = dictionary: keys are species (str), values are arrays of their signals (ndarray)
		N = int: length of signal arrays
		A = float: IRF parameter (ms2/amu)

		Returns:
		y_conv = dictionary: keys are species (str), values are arrays of their signals convolved with IRF (ndarray)
		'''

		# Define convolution functions (h)
		t = np.arange(N)*self._dt	# ms
		h = {}
		for species in m:
			h[species] = np.zeros(N)

			# Leave h(0) = 0 (lim h t-->0 = 0, although directly calculating h(0) is undefined)
			h[species][1:] = np.exp((A * m[species]) / (t[1:])**2) / (t[1:])**4
		
		# Compute convolved signals (y_conv)
		y_conv = {}
		for species in y:
			y_conv[species] = np.zeros(t.size)
		for i in range(t.size):

			for species in y_conv:
				norm_factor = (h[species][:i+1]).sum()

				# Initially, norm_factor = 0 since the t and y arrays are finite and there is a lag between molecules exiting the pinhole and being detected.
				# Therefore, to avoid divide by 0 errors when norm_factor = 0, it makes the most physical sense to set y_conv[species][i] = y[species][0].
				# Once norm_factor != 0, we can start calculating y_conv[species][i] as a convolution.

				if norm_factor == 0.:
					y_conv[species][i] = y[species][0]
				else:
					y_conv[species][i] = (y[species][:i+1] * np.flip(h[species][:i+1],0)).sum() / norm_factor

		return y_conv