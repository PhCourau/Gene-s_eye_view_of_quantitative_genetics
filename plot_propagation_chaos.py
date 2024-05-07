import matplotlib.pyplot as plt
import numpy as np
from simulate_population import Simulate
from PDE_evolution import simulate_PDE,gen_mu0
#This file contains the function plot_propagation_chaos which will plot allele frequency dynamics


N=1000 #Population size
L=100 #Number of loci

theta = (3.3,1.1) #mutation rates (forward, backwards)
s = 10 #strength of selection

T = 1 #duration of a simulation run

def plot_propagation_chaos(list_allele_frequencies,parameters=None,Ny=100):
	"""Plots the result of Simulate.
	Parameters
	----------
	list_allele_frequencies: a (t,L) array giving the evolution of L allele frequencies over a
				time t. Typical imput is the first output of the function Simulate
				(see simulate_population.py)
	parameters: a vector of parameters used to compute theoretical predictions for the evolution
		    of the system. See the third output of Simulate in simulate_population.py. These
		    will be fed to the function simulate_PDE. In particular, the function will assume
		    that the initial distribution is the neutral beta in linkage equilibrium.
	T: optional. Used in the label of the x axis
	Ny: the discretization step in space for the PDE theoretical approximation
	"""
	if parameters is None:
		T=1
		L=L
	else:
		theta,s,N,L,T = parameters
		ytheory = simulate_PDE(gen_mu0(theta,Ny),
					theta,s,T)
		xtheory = np.linspace(0,T,len(ytheory))

	fig=plt.figure()
	ax=plt.axes()
	x = np.linspace(0,T,np.shape(list_allele_frequencies)[0])

	for l in range(L):
		ax.plot(x,list_allele_frequencies[:,l],alpha=.1,color="grey")
	ax.plot(x,
		np.mean(list_allele_frequencies,axis=1),
		color="green")
	if parameters is not None:
		ax.plot(xtheory,ytheory,"orange")
	ax.set_ylim((0,1))
	ax.set_xlabel("t")
	ax.set_ylabel("Frequency of the +1 allele")
	plt.show()
