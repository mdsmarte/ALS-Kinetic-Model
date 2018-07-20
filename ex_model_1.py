# Import modules
import numpy as np
from scipy.integrate import odeint

# Define the user model
def model_H2O2_depletion(t, model_params):
	# First argument is a numpy array of times - evenly spaced in ascending order (ms)
	# Second argument is a dictionary of model parameters: keys are the parameter names 

	# User-defined parameters, X0 must always be included as the initial radical concentration
	# Any parameter that you would like to fit or include in a monte carlo uncertainty simulation must be defined in model_params
	X0 = model_params['X0']					# Initial OH concentration (molc/cm3)
	c_H2O2_0 = model_params['c_H2O2_0']		# Initial H2O2 concentration (molc/cm3)
	k_OH_wall = model_params['k_OH_wall']	# OH wall loss rate (s-1)
	k_HO2_wall = model_params['k_HO2_wall']	# HO2 wall loss rate (s-1)
	k1 = model_params['k1']					# OH + H2O2 --> HO2 + H2O (cm3/molc/s)
	k2 = model_params['k2']					# OH + HO2 --> H2O + O2 (cm3/molc/s)
	k3 = model_params['k3']					# HO2 + HO2 --> H2O2 + O2 (cm3/molc/s)

	# Need to define m and c dictionaries for any species you want the model to output.
	# Any species for which you have data and want to perform a fit are required.
	# Other species are optional, but could be useful if you want to plot them without a fit.

	# Key is species, value is its mass in amu
	m = {}
	m['H2O2']  = 34.005480
	m['OH']    = 17.002740
	m['HO2']   = 32.997655

	# Key is species, value is a numpy array with concentrations at the times in array t
	# Initially, create arrays that are the same size as t with ALL entries being the pre-photolysis (t < 0) concentration
	# We will update these arrays later to contain the modeled concentrations post-photolysis (t >= 0)
	c = {}
	c['H2O2'] = c_H2O2_0*np.ones(t.size)
	c['OH']   = np.zeros(t.size)
	c['HO2']  = np.zeros(t.size)

	# Create the initial concentration (t = 0) array
	# Need to do this for all species that have chemistry in the model, even if they aren't outputted
	H2O2_0 = c_H2O2_0 - X0/2
	OH_0   = X0
	HO2_0  = 0
	H2O_0  = 0
	O2_0   = 0
	y0 = np.array([H2O2_0, OH_0, HO2_0, H2O_0, O2_0])

	# Define the kinetic model
	def calc_dy_dt(y, t_curr):
		# Positions of species in y correspond to the order of species in initial concentration array
		# t_curr is not used since reaction rates depend only on concentrations

		H2O2 = y[0]
		OH   = y[1]
		HO2  = y[2]
		H2O  = y[3]
		O2   = y[4]

		dH2O2 = -k1*OH*H2O2            +k3*HO2*HO2
		dOH   = -k1*OH*H2O2 -k2*OH*HO2               -k_OH_wall*OH
		dHO2  =  k1*OH*H2O2 -k2*OH*HO2 -2*k3*HO2*HO2 -k_HO2_wall*HO2
		dH2O  =  k1*OH*H2O2 +k2*OH*HO2
		dO2   =              k2*OH*HO2 +k3*HO2*HO2

		# Order of species must much the order in the initial concentrations array
		dy_dt = np.array([dH2O2, dOH, dHO2, dH2O, dO2])
		return dy_dt

	# If t[-1] < 0 (all times are pre-photolysis), then no need to integrate the model
	# If t[-1] >= 0 (some times are post-photolysis), then we need to integrate the model and update the concentration arrays
	if t[-1] >= 0:

		# Find the index that corresponds to t = 0 (the below approach is more accurate than == due to numerical roundoff)
		idx_zero = np.abs(t).argmin()

		# Convert ms --> s, then integrate the model over t >= 0
		odeint_out = odeint(calc_dy_dt, y0, t[idx_zero:]/1000)

		# Update the concentration vector over t >= 0, positions correspond to order of species in initial concentration array
		c['H2O2'][idx_zero:] = odeint_out.T[0]
		c['OH'][idx_zero:]   = odeint_out.T[1]
		c['HO2'][idx_zero:]  = odeint_out.T[2]

	return m, c