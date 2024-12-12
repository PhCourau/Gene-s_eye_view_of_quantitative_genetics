import matplotlib.pyplot as plt
import numpy as np
from propagation import Simulate
from PDE_evolution import simulate_PDE,gen_mu0
#This file contains the function plot_propagation_chaos which will plot allele frequency dynamics
#The parameters of the simulation are to be found in the propagation.py file

N=1000 #Population size
L=100 #Number of loci
theta = (1.1,3.3) #mutation rates (forward, backwards)
omega = 15 #strength of selection
T=.3

def plot_propagation_chaos(list_allele_frequencies,Ny=100):
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
	ytheory = simulate_PDE( gen_mu0(theta,Ny),
					theta,omega,T)
	xtheory = np.linspace(0,T*N,len(ytheory))

	fig=plt.figure()
	ax=plt.axes()
	x = np.linspace(0,N*T,np.shape(list_allele_frequencies)[0])

	for l in range(1,L):
		ax.plot(x,list_allele_frequencies[:,l],alpha=.1,color="grey")
	ax.plot(x,
		np.mean(list_allele_frequencies,axis=1),
		color="green")
	ax.plot(xtheory,ytheory,"orange")
	ax.set_ylim((0,1))
	ax.set_xlabel("Generations")
	ax.set_ylabel("Frequency of the +1 allele")
	plt.show()

def Figure_propchaos():
	list_allele_frequencies = Simulate(T,theta,omega,L,N)
	plot_propagation_chaos(list_allele_frequencies)
