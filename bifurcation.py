import matplotlib.pyplot as plt
import numpy as np
from simulate_population import Population
from simulate_population import generate_pop
import scipy.special as special

#This file contains the function plot_Figurebif which draws the bifurcation diagram for
#stabilising/disruptive selection.


N = 200 #Population size
L = 100
theta = (.6,.6) #mutation rates (forward, backwards)
kapparange = (-5,0,40) #range of simulated kappa values
kapparangeanalytic = (-5,0,1000) #range of computed s values
T= 10

invphi = (np.sqrt(5)-1)/2

def mean_phenotype(theta,s):
	"""Finds the mean phenotype associated with a modified beta solution
	to a WF with selection s"""
	return (2*theta[0]/(theta[0]+theta[1])
		*special.hyp1f1(2*theta[0]+1,2*(theta[0]+theta[1])+1,2*s)
		/special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*s)
		-1)


def match_equation(Z,s,theta):
	"""Checks the distance between Z and the mean of pi_Z (see Theorem 5.1 of our paper)"""
	return np.abs(mean_phenotype(theta,-2*Z*s)- Z)



#Here we simulate values of equilibria by letting the system evolve for a time T at each value of s.
#N=10000

def simulate_equilibria(kapparange = kapparange,theta=theta,T=T,N=N,L=L):
	"""This function computes for each s in kapparange the +1 allele frequency after time T twice:
	once if we start from all +1, the other if we start from all -1"""
	equilibria =[0]*2*kapparange[-1]
	for (kappaindex,kappa) in enumerate(np.linspace(*kapparange)):
		pop=Population(theta,
				N,
				L,
				population= np.array([[False]*L]*N,dtype='bool'),
				alpha = np.ones(L))
		for t in range(int(N*T)):
			pop.selection_drift_sex(kappa,0,N,L)
			pop.mutation(theta,N,L)
		equilibria[2*kappaindex] = 2*np.mean(pop.population)-1
		pop=Population(theta,
				N,
				L,
				population=np.array([[True]*L]*N,dtype='bool'),
				alpha = np.ones(L))
		for t in range(int(N*T)):
			pop.selection_drift_sex(kappa,0,N,L)
			pop.mutation(theta,N,L)
		equilibria[2*kappaindex+1] = 2*np.mean(pop.population)-1
		print("Done kappa="+str(kappa))
	return np.array(equilibria)




# Here we find the equilibria with a golden search algorithm and plot simulations versus analytic predictions
def gss(f, a, b, tolerance=1e-5):
	"""
	Golden-section search
	to find the minimum of f on [a,b]

	* f: a strictly unimodal function on [a,b]

	Example:
	>>> def f(x): return (x - 2) ** 2
	>>> x = gss(f, 1, 5)
	>>> print(f"{x:.5f}")
	2.00000

	Credit to Wikipedia
	"""
	invphi = (np.sqrt(5) - 1) / 2  # 1 / phi (inverse golden ratio)
	while b - a > tolerance:
		c = b - (b - a) * invphi
		d = a + (b - a) * invphi
		if f(c) < f(d):
			b = d
		else:  # f(c) > f(d) to find the maximum
			a = c
	return (b + a) / 2


def plot_Figurebif(kapparange=kapparange,kapparangeanalytic=kapparangeanalytic,theta=.6,T=T,N=N,L=L):
	"""Plots our desired figure.
	Parameters:
	-----------
	kapparange: a triplet (smin,smax,nbstep) to be fed into np.linspace to get all tested values of s.

	kapparangeanalytic: a triplet to get all values of s for analytical predictions

	theta: a single positive float (symmetric mutation rate)

	T: how long to simulate a population before we consider that it is at equilibrium

	equilibria: if None, empirical equilibria are simulated using the function simulate_equilibria
		    Otherwise, imput a list of 2*nbsteps arrays, for instance [simulate_equilibria()]

	"""
	equilibria = [simulate_equilibria(kapparange = kapparange,theta=(theta,theta),T=T,N=N,L=L)]
	fig = plt.figure()
	ax=plt.axes()
	for eq in equilibria:
		ax.plot(np.repeat(np.linspace(*kapparange),2),eq,"or")

	#Plot central branch
	kappa_c = (4*theta+1)/2
	ax.plot([-kappa_c,kapparange[0]],[0,0],"blue",linestyle="--")

	list_kappa = np.linspace(*kapparangeanalytic)
	#Top branch
	topbranch = np.zeros(kapparangeanalytic[2])
	for (k,s) in enumerate(list_kappa):
		topbranch[k] = gss(lambda z: match_equation(z,s,(theta,theta)),0,1)
	ax.plot(list_kappa,topbranch,"blue")

	#Bottom branch
	bottombranch = np.zeros(kapparangeanalytic[2])
	for (k,s) in enumerate(list_kappa):
		bottombranch[k] = gss(lambda z: match_equation(z,s,(theta,theta)),-1,0)
	ax.plot(list_kappa,bottombranch,"blue")


	ax.set_xlabel("$\kappa$")
	ax.set_ylabel("$\mathbb{E}[2f_t-1]$")
	plt.show()



#An accessory function for another cool plot.

def ismatched_equation(Z,s,theta,epsilon):
	"""Checks if a given Z value satisfies the nonlinear equation"""
	if np.abs(np.mean_phenotype(theta,-2*s*Z) -Z) <= epsilon:
		return True
	return False

def find_solutions(theta,epsilon=1e-3):
	list_kappa = np.linspace(-15,1,200)
	list_Z = np.linspace(-1,1,1001)
	list_points = []
	for s in list_kappa:
		for Z in list_Z:
			if ismatched_equation(Z,s,theta,epsilon):
				list_points.append([s,Z])
	return np.transpose(np.array(list_points))


def plot_bifurcations(  list_theta = (.6,.6),
			N=N,
			kapparange=kapparange,
			nbsteps=1000):
	"""Plots a bifurcation diagram showing the fixed points (s,Z) for different values of theta
	Parameters:
	-----------
	"""
	ax = plt.axes()
	points = find_solutions(theta,N=N,kapparange=kapparange,nbsteps=nbsteps)
	ax.plot(points[0],
		points[1],
		"o",
		color="blue")
		#label="θ1="+str(theta[0])+", θ2="+str(theta[1]))
	ax.set_ylim((-1,1))
	ax.set_xlabel("ω")
	ax.set_ylabel("s*")
	ax.set_title()
	#ax.legend(handles = list_plots)
	plt.show()

