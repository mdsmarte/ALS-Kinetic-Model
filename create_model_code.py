# Run from the command line
# Add API
# Add to docs the checks that it performs for proper formatting (reactants and products empty being handled differently)
# See provided input and output files '' and '' for an example

import csv

def create_model_code(file_in, file_out, rxn_delim='==>'):

	k_list = []
	dy_dt = {}

	with open(file_in) as f:

		reader = csv.reader(f)
		for i, rxn in enumerate(reader):

			if len(rxn) != 2:
				raise ValueError('Input file line {:d} is not properly comma separated!'.format(i+1))

			# Add rate constant to global rate constant list
			k = rxn[0]
			if k in k_list:
				raise ValueError('Rate constant \'{}\' appears in input file more than once!'.format(k))
			k_list.append(k)

			# Break apart the reaction line
			rxn_split = rxn[1].split(rxn_delim)
			if len(rxn_split) != 2:
				raise ValueError('Reaction on line {:d} of input file is not properly formatted!'.format(i+1))
			left = rxn_split[0].split('+')
			right = rxn_split[1].split('+')
			if left == ['']:
				raise ValueError('No reactants specified for the reaction on line {:d} of input file!'.format(i+1))

			# Process stoichiometry to create reactant and product lists
			# Example: 2A ==> B + 2C should produce reactants = ['A','A'] and products = ['B','C','C']
			def process_coeff(x):
				idx = 0
				while x[:idx+1].isdigit():
					idx += 1
				return ([x] if idx == 0 else int(x[:idx])*[x[idx:]])

			reactants = []
			for term in left:
				reactants += process_coeff(term)
			products = []
			for term in right:
				products += process_coeff(term)

			# Add a key in dy_dt for any new species
			for species in (reactants+products):
				if species not in dy_dt.keys():
					dy_dt[species] = []

			# Add reaction rate terms to diff eq matrix
			rxn_txt = (k + '*' + '*'.join(reactants))

			for species in reactants:
				dy_dt[species].append('-' + rxn_text)
			for species in products:
				dy_dt[species].append('+' + rxn_text)

	species_names = dy_dt.keys()

	with open(file_out, 'w') as f:

		# print(file=f)
		# Check out the docs for the *objects and kwargs, could make code cleaner, especially sep param

		pass
