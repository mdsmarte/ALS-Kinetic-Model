# Author: Matthew Smarte
# Version:
# Date:

# List of dependencies

import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from IPython.display import display

# TODO:
# Add checks to make sure the inputs from the user are correct, throw errors if they are not
# Try to stick to the PEP 8 python style recommendation
# Add save output options
# Figure out best way to create tables, plots in the notebook - from IPython.display import display

class KineticModel:

	_dt = 0.02	# Fundamental kinetic time step (ms)

	def __init__(self, user_model):
		self._user_model = user_model
		# Add code to verify the structure of the model agrees with formatting requirements

	def fit(self, t, tbin, data, model_params, ALS_params, err_weight=True, fit_pre_photo=False, apply_IRF=True, apply_PG=True, t_PG=1, plot_fits=True):#, save=False, filename=None):
		'''
		Input:   time axis, species data, params - initial guesses as well as fixed paremters, flags for using IRF and photolysis gradient,
		         flags for printing the output and plotting the fits, full output or abbreviated output
		Returns: fitted parameters and their uncertainties
		         if full output is specified, also return covariance matrix, correlation matrix, cost function value, leastsq mesg and ier

		t is the time axis for the data values and errors
		the smallest point in t can never be less than -20 ms
		'''
		
		# Make sure there are no zeros in the error arrays (will lead to divide by zero errors)
		# Be sure to have comments on how weighting works
		# Add code to verify the structure of the data and params agrees with formatting requirements - should checks of model_params and ALS_params be functions?
		if ALS_params.at['t0','fit'] and not fit_pre_photo:
			# Throw error - if t0 is being fit then, fit_pre_photo must be True
			pass

		# Determine start time, end time, and range of times over which to fit
		t_start = t.min()
		t_end = t.max()
		idx_data = np.full(t.shape, True) if fit_pre_photo else (t >= ALS_params.at['t0','val'])	# Boolean array

		# Organize fitted species data into data_val and data_err frames
		# Columns of data_val and data_err are species and the rows correspond to times in t array
		data_fit = data[data['fit']]
		species_names = list(data_fit.index)
		data_val = pd.DataFrame(list(data_fit['val']), index=species_names).T
		if err_weight:
			data_err = pd.DataFrame(list(data_fit['err']), index=species_names).T

		# Organize the fit parameters
		model_params_fit = model_params[model_params['fit']] 
		ALS_params_fit = ALS_params[ALS_params['fit']]
		p_names = list(model_params_fit.index) + list(ALS_params_fit.index)
		p0 = np.concatenate((model_params_fit['val'], ALS_params_fit['val']))

		# Establish corrspondence between data and model time axes
		# Each entry in t has an exact match to an entry in t_model
		# We take the below approach (rather than ==) to prevent any problems with numerical roundoff
		t_model = self._time_axis(t_start, t_end, tbin)
		idx_model = [np.abs(t_model - t[idx_data][_]).argmin() for _ in range(sum(idx_data))]	# Array of positional indices

		# Define the cost function to be optimized
		def calc_cost(p):

			# Organize parameter values used for the current iteration of the fit into dictionaries
			model_params_p = {}
			for param in model_params.index:
				model_params_p[param] = p[p_names.index(param)] if param in p_names else model_params.at[param,'val']
			ALS_params_p = {}
			for param in ALS_params.index:
				ALS_params_p[param] = p[p_names.index(param)] if param in p_names else ALS_params.at[param,'val']

			# Run the model - we only need the concentrations dataframe
			c_model = self._model(t_start, t_end, tbin, model_params_p, ALS_params_p, apply_IRF, apply_PG, t_PG)[1]

			# Calculate the weighted residual array
			res = []
			for species in species_names:
				# Remember: idx_data is a boolean array and idx_model is an array of positional indices
				obs = data_val.loc[idx_data,species]
				mod = ALS_params_p['S_'+species] * c_model.iloc[idx_model,c_model.columns.get_loc(species)]

				# We take the sqrt of the weight since leastsq will square the array later
				species_res = np.sqrt(data_fit.at[species,'weight']) * (obs-mod).values

				if err_weight:
					err = data_err.loc[idx_data, species]
					species_res = species_res / err.values

				res.append(species_res)

			# leastsq will square then sum all entries in the returned array, and minimize this cost value
			return np.concatenate(res)

		# Perform the fit
		# NOTE: The backend of leastsq will automatically autoscale the fit parameters to the same order of magnitude if diag=None (default).
		p, cov_p, infodict, mesg, ier = leastsq(calc_cost, p0, full_output=1)

		# Calculate minimized cost value
		cost = np.sum(infodict['fvec']**2)

		# Prepare covariance and correlation matrices
		if cov_p is not None:
			# NEED TO REVIST THE LOGIC OF THIS NOW THAT WE HAVE ADDED SPECIES WEIGHTING FUNCTIONALITY

			# If cost function is not error weighted, then scale the covariance matrix according to documentation for leastsq and source code for curve_fit.
			# Scale factor is cost / (# of data points - # fit parameters)
			if not err_weight:
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

		# Display results
		print('Optimization terminated successfully.' if ier in (1,2,3,4) else 'Optimization FAILED.')
		print('Exit Code = {:d}'.format(ier))
		print('Exit Message = {}'.format(mesg))
		print()

		print('Optimized Cost Function Value = {:g}'.format(cost))
		print()

		print('Optimized Parameters and Standard Errors:')
		display(df_p)
		print()

		print('Correlation Matrix:')
		display(df_corr_p)
		print()

		# Plot the fits
		if plot_fits:
			pass

		return df_p, df_cov_p, df_corr_p, cost, mesg, ier


	# Should we add save functionality to this? Save the scaled model output
	# Is plot_fit the correct name?
	def plot_fit(self, t, tbin, data, model_params, ALS_params, apply_IRF=True, apply_PG=True, t_PG=1):
		'''
		Plots the model overlaid on the inputted species data with residuals
		Only plots species for which fit=True in the data dataframe
		'''

		# Need to handle the case where nSpecies is < 3

		# Run the model
		t_start = t.min()
		t_end = t.max()
		t_model, c_model = self.run(t_start, t_end, tbin, model_params, ALS_params, apply_IRF, apply_PG, t_PG)

		# Only plot the species for which fit=True
		# Columns of data_val are species and the rows correspond to times in t array
		data_fit = data[data['fit']]
		species_names = list(data_fit.index)
		nSpecies = len(species_names)
		data_val = pd.DataFrame(list(data_fit['val']), index=species_names).T

		f, ax = plt.subplots(2, nSpecies, gridspec_kw={'height_ratios':[3, 1]})
		if nSpecies == 1:
			ax = ax.reshape((2,1))

		'''
		plt.rcParams['font.family'] = 'serif'
		plt.rcParams['font.serif'] = 'Ubuntu'
		plt.rcParams['font.monospace'] = 'Ubuntu Mono'
		plt.rcParams['font.size'] = 10
		plt.rcParams['axes.labelsize'] = 10
		plt.rcParams['axes.labelweight'] = 'bold'
		plt.rcParams['xtick.labelsize'] = 8
		plt.rcParams['ytick.labelsize'] = 8
		plt.rcParams['legend.fontsize'] = 10
		plt.rcParams['figure.titlesize'] = 12
		'''

		for i, species in enumerate(species_names):
			fit = ALS_params.loc['S_'+species,'val']*c_model[species]

			ax[0,i].set_title(species)					# Make the title the species name
			ax[0,i].plot(t, data_val[species], 'o')		# Plot the data
			ax[0,i].plot(t, fit, linewidth=2.) 			# Plot the fit
			ax[1,i].plot(t, data_val[species]-fit, 'o')	# Plot residual
			ax[1,i].plot(t, np.zeros(t.shape))			# Plot zero residual line

		ax[0,0].set_ylabel('Data & Fit')
		ax[1,0].set_ylabel('Data - Fit')
		#ax[1,1].set_xlabel('Time (ms)')


		plt.show()


	def run(self, t_start, t_end, tbin, model_params, ALS_params, apply_IRF=True, apply_PG=True, t_PG=1): #, plot_results=True, save=False, filename=''):
		'''
		# Inputs:  time axis, params - all fixed, species for which we want output, flags for plotting
		# Returns: concentrations as a function of time for each of the species
		# Only uses 'val' column of parameters

		# t_start and t_end need to be integer multiples of dt*tbin and t_PG
		'''

		model_params_val = model_params['val'].to_dict()
		ALS_params_val = ALS_params['val'].to_dict()

		t_model, c_model = self._model(t_start, t_end, tbin, model_params_val, ALS_params_val, apply_IRF, apply_PG, t_PG)

		return t_model, c_model

	def plot_model(self):
		'''
		Plots the model without the data
		'''
		pass
	
	def bootstrap(self):
		'''
		If fit_pre_photo is True then 
		'''
		# Be sure to make a copy of the data frame so it doesn't get overwritten

		pass

	def monte_carlo(self):
		'''
		Need to make sure parameters don't become nonphysical (e.g. negative rate constants) when simulating
		'''
		# Be sure to make a copy of the params data frames so that they don't get overwritten


		pass

	def _time_axis(self, t_start, t_end, tbin):
		return np.linspace(t_start, t_end, num=int(((t_end-t_start)/(tbin*self._dt))+1), endpoint=True)

	def _model(self, t_start, t_end, tbin, model_params, ALS_params, apply_IRF, apply_PG, t_PG):
		# model_params and ALS_params are now dictionaries
		# Only A, B, and t0 are used from ALS_params - the sensitivity parameters are ignored
		# A copy is created for anything passed to the user model to prevent problems with any mutable objects

		# Wrapper function for running various models, implementing photolysis gradient and/or IRF
		# Models must take initial radical concentration as a variable, which is updated based on the photolysis gradient
		# X0 is the initial radical concentration
		# args are additional optional arguments to pass to the model function
		# kwargs contaings flags for using photolysis gradient or IRF
		# add photolysis gradient parameter later

		# t_start, t_end, t0, tbin_PG must always be integer multiples of 0.02 ms
		# Minimum value of t_start is -20 ms
		# t_start and t_end must be integer multiples of t_bin*dt
		# t_end must be > 0 and t_start <= 0

		# Create time axis for running the model (before applying tbin and t0)
		t0 = ALS_params['t0']
		if t0 == 0:
			t = self._time_axis(-20, t_end+((tbin-1)*self._dt), 1)
		elif t0 < 0:
			t = self._time_axis(-20, t_end+((tbin-1)*self._dt)-t0, 1)
		elif t0 > 0:
			t = self._time_axis(-20-t0, t_end+((tbin-1)*self._dt), 1)
		
		# Model integration w/ photolysis gradient
		if apply_PG:

			X0 = model_params['X0']
			B = ALS_params['B']
			idx_zero = np.abs(t).argmin()

			# First populate concentration matrix for t < 0
			m, c = self._user_model(t[:idx_zero].copy(), model_params.copy())

			# Then populate concentration matrix for t >= 0
			idx_curr = idx_zero
			idx_step = int(t_PG/self._dt)

			while idx_curr < t.size:
				t_mid = t[idx_curr] + t_PG/2
				X_curr = X0*(1+B*t_mid)

				model_params_curr = model_params.copy()
				model_params_curr['X0'] = X_curr

				c_tmp = self._user_model(t[:idx_curr+idx_step].copy(), model_params_curr)[1]

				for species in c:
					c[species] = np.append(c[species], c_tmp[species][idx_curr:])

				idx_curr += idx_step	

		# Model integration w/o photolysis gradient
		else:
			m, c = self._user_model(t.copy(), model_params.copy())

		# Convolve with IRF
		if apply_IRF:
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
		Convolves signals with the mass-dependent ALS instrument response function.

		At the ALS, molecules exit the pinhole with a Maxwell-Boltzmann distribution of velocities.
		The ion signal at a specific observation time therefore arises from molecules that exit the
		pinhole at a distribution of kinetic times that all come before the observation time.  See
		more details here: https://doi.org/10.1002/kin.20262

		Each convolved point is thus calculated as a weighted average of all points in the model at 
		earlier times, where the weighting is given by the mass-dependent IRF function.

		Note that this operation is not quite a "convolution" as one might usually think about it,
		since a convolved point only contains contributions from the modeled points at earlier times,
		and has zero contribution from modeled points at later times.

		Inputs:
		m = dictionary: keys are species, values are their masses (amu)
		y = dictionary: keys are species, values are arrays of their signals
		N = int: length of signal arrays
		A = float: IRF parameter (amu-1)

		Outputs:
		y_conv = dictionary: keys are species, values are arrays of their signals convolved with IRF
		'''

		# Define convolution functions (h)
		t = np.arange(N)*self._dt
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
				norm_factor = np.sum(h[species][:i+1])

				# Initially, norm_factor = 0 since the t and y arrays are finite and there is a lag between molecules exiting the pinhole and being detected.
				# Therefore, to avoid divide by 0 errors when norm_factor = 0, it makes the most physical sense to set y_conv[species][i] = y[species][0].
				# Once norm_factor != 0, we can start calculating y_conv[species][i] as a convolution.

				if norm_factor == 0.:
					y_conv[species][i] = y[species][0]
				else:
					y_conv[species][i] = np.sum(y[species][:i+1] * np.flip(h[species][:i+1],0)) / norm_factor

		return y_conv