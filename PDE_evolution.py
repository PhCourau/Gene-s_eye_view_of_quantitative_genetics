#This file is concerned with modelling the solution to the following PDE on [0,1]:
# d/dt \mu(x) = -d/dx [a(x,\mu)  \mu] + 1/2  d**2/(dx)**2 (x(1-x) \mu)
#with
#	a(x,\mu )= -s\int(2x'-1) \mu(dx')x(1-x) + (alpha (1-x) - beta x)
#
# This is a tricky equation to simulate. Fortunately the following paper has a good approximation:
#https://www.researchgate.net/profile/Lin-Chen-147/publication/366488223_Strong_convergence_and_stationary_distribution_of_an_explicit_scheme_for_the_Wright-Fisher_model/links/63cd365e6fe15d6a573dc5c8/Strong-convergence-and-stationary-distribution-of-an-explicit-scheme-for-the-Wright-Fisher-model.pdf
# We will use the same notations as in that article, with the additionnal s and theta:=(alpha,beta)

#We use the following Lamperti transform:
#F(x) = 2arcsin(sqrt(x))
#F has inverse invF(y) = sin(x/2)**2
#
# If we make the change in variables y=phi(x), then the PDE becomes a PDE on [-pi,pi]
#d/dt \mu(y) = -d/dy [b(x,\mu) \mu] + 1/2 d**2/(dy)**2 \mu
#with
#	b(y,\mu) = -s c(y,\mu) sin(y/2) cos(y/2) + (theta[0]-1/4)cotan(y/2)-(theta[1]-1/4)tan(y/2)
#with
#	c(y,\mu) = \int (2psi(y)-1) psi'(y)\mu(dy)
#
# In particular, in this new PDE \mu(-pi) = \mu(pi) = 0
import numpy as np
import matplotlib.pyplot as plt

#Attempt at a direct Fokker-Planck
def gen_mu0(theta,N):
	x=np.linspace(0,1,N)
	pi= x**(2*theta[0]-1) * (1-x)**(2*theta[1]-1)
	return pi/np.sum(pi)

def FP_WF(mu,theta,s,dt):
	x = np.linspace(0,1,len(mu))
	dy = 1/len(mu)
	selection = -2*s*(2*np.sum(x*mu)-1)
	func_to_derive = mu*(theta[0]*(1-x) - theta[1]*x + x*(1-x)*selection)

	Derivative = (func_to_derive[1:-1]-func_to_derive[:-2])/dy

	Laplacian = ((mu*x*(1-x))[:-2] + (mu*x*(1-x))[2:]-2*(mu*x*(1-x))[1:-1])/(dy)**2

	mu[1:-1] = mu[1:-1] + dt*(-Derivative + Laplacian/2)
	mu = np.max([mu,[0]*len(mu)],axis=0)
#	if np.min(mu)<dt:
		#print("pb") #raise Warning("Runtime warning: numerical instabilities have appeared. Try a smaller value for dt")
	return mu/np.sum(mu)

def simulate_PDE(mu0,theta,s,T,dt=None):
	"""Simulates the solution to the PDE and returns a vector for the evolution of the mean
	Parameters:
	-----------
	mu0
	theta
	s
	N
	T"""
	N=len(mu0)
	x=np.linspace(0,1,N)
	if dt is None:
		dt=1/(N**2)
	list_means = [0]*int(T/dt)
	for t in range(int(T/dt)):
		mu0 = FP_WF(mu0,theta,s,dt)
		list_means[t] = np.sum(x*mu0)
		if t%(int(T/dt/100)) == 0:
			print("Done step " + str(t) + " of "+str(T/dt))
	return list_means




#All this is crap (probably) to try and simulate FP with theta<1
def F(x):
	"""See equation (6)"""
	return 2*np.arcsin(np.sqrt(x))

def invF(y):
	"""See equation (7)"""
	return np.sin(y/2)**2

def dinvF(y):
	"""The derivative of invF"""
	return np.cos(y/2)*np.sin(y/2)

def meanmu(mu):
	"""Returns the mean of a distribution mu in y, transformed to be in x"""
	y = np.linspace(0,np.pi,len(mu))
	return np.sum(invF(y[1:-1])* mu[1:-1])

def generate_mu0(theta=(.6,1.2),N=100):
	"""Test function to generate a dummy mu"""
	y = np.linspace(0,np.pi,N)
	mu = invF(y)**(2*theta[0]-1) * (1-invF(y))**(2*theta[1]-1) * dinvF(y)
	mu[0],mu[-1] = 0,0
	return mu/np.sum(mu)

def f(y,s,c,alpha,beta):
	"""See equation (9)"""
	return -s*c*np.cos(y/2)*np.sin(y/2) + (alpha-1/4)/np.tan(y/2) - (beta-1/4)*np.tan(y/2)

def df(y,s,c,alpha,beta):
	"""See equation (17)"""
	return (-s*c*(np.cos(y/2)**2-1/2)
		+ (alpha-1/4)/(2*np.sin(y/2)**2) - (beta-1/4)/(2*np.cos(y/2)**2))

def fdelta(y,deltak,s,c,alpha,beta):
	"""See equation (21)"""
	C0 = (alpha + beta - 1/2)/2
	return np.where(
		y> np.pi,
		(f(np.pi-deltak,s,c,alpha,beta) + 1/2*deltak*df(np.pi-deltak,s,c,alpha,beta)
		 - C0*(y-np.pi + 1/2*deltak)),
		0
	) + np.where(
		(y<=np.pi) * (y>np.pi-deltak),
		(f(np.pi-deltak,s,c,alpha,beta) + df(np.pi-deltak,s,c,alpha,beta)*(y-np.pi+deltak)
		- 1/(2*deltak)*(df(np.pi-deltak,s,c,alpha,beta)+C0)*(y-np.pi+deltak)**2),
		0
	) + np.where(
		(y<=np.pi-deltak)*(y>deltak),
		f(y,s,c,alpha,beta),
		0
	) + np.where(
		(y<=deltak) * (y>=0),
		(f(deltak,s,c,alpha,beta) + df(deltak,s,c,alpha,beta)*(y-deltak)
		- 1/(2*deltak)*(df(deltak,s,c,alpha,beta)+C0)*(y-deltak)**2),
		0
	) + np.where(
		y<0,
		(f(deltak,s,c,alpha,beta) + 1/2*deltak*df(deltak,s,c,alpha,beta)
		 - C0*(y- 1/2*deltak)),
		0)




def iterate(mu,theta,s,dt):
	"""Simulates one step of time in the PDE
	Parameters:
	----------
	mu: a vector, discretization of the measure (with spacestep dy)
	theta,s: mutation and selection
	dt: timestep. Should be smaller than dy**2
	See equation (24)
	"""
	sum_theta = theta[0]+theta[1]
	N = len(mu)
	dy = np.pi/N #spatial scale
	y = np.linspace(0,np.pi,N)

	#first term
	c = 2*meanmu(mu)-1
	b = f(y[1:-1],s,c,theta[0],theta[1])
	func_to_derive = np.copy(mu)
	func_to_derive[1:-1] = func_to_derive[1:-1]* b
	Derivative = (func_to_derive[1:-1]-func_to_derive[:-2])/dy

	#Second term
	Laplacian = (mu[:-2] + mu[2:]-2*mu[1:-1])/(dy)**2


	mu[1:-1] = mu[1:-1] + dt*(-Derivative + Laplacian/2)
	if np.min(mu)<0:
		#plt.plot(mu);plt.show()
		raise Warning("Runtime warning: numerical instabilities have appeared. Try a smaller value for dt")
	return mu/np.sum(mu)

def simulate_PDEcrap(mu0,theta,s,T,dt=None):
	"""Simulates the solution to the PDE and returns a vector for the evolution of the mean
	Parameters:
	-----------
	mu0
	theta
	s
	N
	T"""
	N=len(mu0)
	if dt is None:
		dt=1/(10*N**2)
	list_means = [0]*int(T/dt)
	for t in range(int(T/dt)):
		mu0 = iterate(mu0,theta,s,dt)
		list_means[t] = meanmu(mu0)
		if t%(int(T/dt/100)) == 0:
			print("Done step " + str(t) + " of "+str(T/dt))
			#plt.plot(mu0);plt.show()
	return list_means,mu0


