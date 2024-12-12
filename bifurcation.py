import matplotlib.pyplot as plt
import numpy as np
from simulate_population import Population
from simulate_population import generate_pop
import scipy.special as special

#This file contains the function plot_Figurebif which draws the bifurcation diagram for 
#stabilising/disruptive selection.


N = 200 #Population size
L = 100
theta = (.6,.6) #(.7,.6) #mutation rates (forward, backwards) (.6,.9)
srange = (-5,0,40) #range of simulated s values
srangeanalytic = (-5,0,1000) #range of computed s values
T= 10

invphi = (np.sqrt(5)-1)/2

def mean_phenotype(theta,s):
	"""Finds the mean phenotype associated with a modified beta solution
	to a WF with selection s"""
	return (2*theta[0]/(theta[0]+theta[1])
		*special.hyp1f1(2*theta[0]+1,2*(theta[0]+theta[1])+1,2*s)
		/special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*s)
		-1)


def match_equation(Z,s,theta,nbsteps=1000):
	"""Checks the distance between Z and the mean of pi_Z (see Theorem 5.1 of our paper)"""
	return np.abs(mean_phenotype(theta,-2*Z*s)- Z)



#Here we simulate values of equilibria by letting the system evolve for a time T at each value of s.
#N=10000

def simulate_equilibria(srange = srange,theta=theta,T=T,N=N,L=L):
	"""This function computes for each s in srange the +1 allele frequency after time T twice:
	once if we start from all +1, the other if we start from all -1"""
	equilibria =[0]*2*srange[-1]
	for (sindex,s) in enumerate(np.linspace(*srange)):
		pop=Population(theta,
				N,
				L,
				population= np.array([[False]*L]*N,dtype='bool'),
				alpha = np.ones(L))
		for t in range(int(N*T)):
			pop.selection_drift_sex(s,0,N,L)
			pop.mutation(theta,N,L)
		equilibria[2*sindex] = 2*np.mean(pop.population)-1
		pop=Population(theta,
				N,
				L,
				population=np.array([[True]*L]*N,dtype='bool'),
				alpha = np.ones(L))
		for t in range(int(N*T)):
			pop.selection_drift_sex(s,0,N,L)
			pop.mutation(theta,N,L)
		equilibria[2*sindex+1] = 2*np.mean(pop.population)-1
		print("Done s="+str(s))
	return np.array(equilibria)




def plot_Figurebifcolor(srange=srange,theta=theta,T=T,N=N,L=L,equilibria=None):
	"""Plots our desired figure.
	Parameters:
	-----------
	srange: a triplet (smin,smax,nbstep) to be fed into np.linspace to get all tested values of s.

	theta: a tuple of positive floats for the mutation rates (forward, backward).

	T: how long to simulate a population before we consider that it is at equilibrium

	equilibria: if None, empirical equilibria are simulated using the function simulate_equilibria
		    Otherwise, imput a list of 2*nbsteps arrays, for instance [simulate_equilibria()]

	"""
	if equilibria is None:
		equilibria = [simulate_equilibria(srange = srange,theta=theta,T=T,N=N,L=L)]
	fig = plt.figure()
	ax=plt.axes()
	for eq in equilibria:
		ax.plot(np.repeat(np.linspace(*srange),2),eq,"or")

	list_s = np.linspace(*srangeanalytic)
	list_Z = np.linspace(-1,1,2*N)
	data = np.zeros((2*N,srangeanalytic[-1]))
	for (inds,s) in enumerate(list_s):
		for (indZ,Z) in enumerate(list_Z):
			data[indZ,inds] = -np.log(match_equation(Z,s,theta))

	CS = ax.contourf(list_s,list_Z,data,levels=100)
	fig.colorbar(CS)

	ax.set_xlabel("ω")
	ax.set_ylabel("s")
	ax.set_title(
"Equilibrium value of the mean phenotype for stabilising/disruptive selection")
	plt.show()


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

	"""
	invphi
	while b - a > tolerance:
		c = b - (b - a) * invphi
		d = a + (b - a) * invphi
		if f(c) < f(d):
			b = d
		else:  # f(c) > f(d) to find the maximum
			a = c
	return (b + a) / 2


def plot_Figurebif(srange=srange,srangeanalytic=srangeanalytic,theta=.6,T=T,N=N,L=L):
	"""Plots our desired figure.
	Parameters:
	-----------
	srange: a triplet (smin,smax,nbstep) to be fed into np.linspace to get all tested values of s.

	srangeanalytic: a triplet to get all values of s for analytical predictions

	theta: a single positive float (symmetric mutation rate)

	T: how long to simulate a population before we consider that it is at equilibrium

	equilibria: if None, empirical equilibria are simulated using the function simulate_equilibria
		    Otherwise, imput a list of 2*nbsteps arrays, for instance [simulate_equilibria()]

	"""
	equilibria = [simulate_equilibria(srange = srange,theta=(theta,theta),T=T,N=N,L=L)]
	fig = plt.figure()
	ax=plt.axes()
	for eq in equilibria:
		ax.plot(np.repeat(np.linspace(*srange),2),eq,"or")

	#Plot central branch
	ax.plot([srange[0],srange[1]],[0,0],"blue")

	list_s = np.linspace(*srangeanalytic)
	#Top branch
	topbranch = np.zeros(srangeanalytic[2])
	for (k,s) in enumerate(list_s):
		topbranch[k] = gss(lambda z: match_equation(z,s,(theta,theta)),0,1)
	ax.plot(list_s,topbranch,"blue")

	#Bottom branch
	bottombranch = np.zeros(srangeanalytic[2])
	for (k,s) in enumerate(list_s):
		bottombranch[k] = gss(lambda z: match_equation(z,s,(theta,theta)),-1,0)
	ax.plot(list_s,bottombranch,"blue")


	ax.set_xlabel("ω")
	ax.set_ylabel("s")
	plt.show()



#An accessory function for another cool plot.

def ismatched_equation(Z,s,theta,epsilon):
	"""Checks if a given Z value satisfies the nonlinear equation"""
	if np.abs(np.mean_phenotype(theta,-2*s*Z) -Z) <= epsilon:
		return True
	return False

def find_solutions(theta,epsilon=1e-3):
	list_s = np.linspace(-15,1,200)
	list_Z = np.linspace(-1,1,1001)
	list_points = []
	for s in list_s:
		for Z in list_Z:
			if ismatched_equation(Z,s,theta,epsilon):
				list_points.append([s,Z])
	return np.transpose(np.array(list_points))


def plot_bifurcations(  list_theta = (.6,.6),
			N=N,
			srange=srange,
			nbsteps=1000):
	"""Plots a bifurcation diagram showing the fixed points (s,Z) for different values of theta
	Parameters:
	-----------
	"""
	ax = plt.axes()
	points = find_solutions(theta,N=N,srange=srange,nbsteps=nbsteps)
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

