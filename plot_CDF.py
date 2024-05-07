import matplotlib.pyplot as plt
import numpy as np
N=1000
s=10
theta=(.1,.3) #(.6,1.2)
T=10

# Here we define the function plotCDFvstheory which plots the cumulative density function of
# a population at time 0 and T, and the theoretical predictions associated
def getCDF(listp):
        """Plots the cumulative distribution function of a vector of frequencies of size L"""
        L=np.shape(listp)[0]
        listy = np.repeat(np.array([k/L for k in range(L+1)]),2)
        listx = np.repeat(np.sort(listp),2)
        listx = np.append(listx,1)
        listx = np.append(0,listx)
        return (listx,listy)

def sqZEZ(Z,s,theta,N):
	"""Computes the squared difference between Z and piZ where pi is a modified beta"""
	x = np.linspace(1/(2*N),1-1/(2*N),N)
	pi = x**(2*theta[0]-1) * (1-x)**(2*theta[1]-1) * np.exp(-4*s*Z*x)
	pi/= np.sum(pi)
	EZ = np.sum(pi*(2*x-1))
	return (Z-EZ)**2


def gss(f, a=-1, b=1, tol=1e-5):
	"""Golden-section search
	to find the minimum of f on [a,b]
	f: a strictly unimodal function on [a,b]

	Example:
	>>> f = lambda x: (x - 2) ** 2
	>>> x = gss(f, 1, 5)
	>>> print("%.15f" % x)
	2.000009644875678

	This code was based on the Wikipedia article on Golden-section article
	"""
	gr = (1+np.sqrt(5))/2 #golden ratio
	while abs(b - a) > tol:
		c = b - (b - a) / gr
		d = a + (b - a) / gr
		if f(c) < f(d):  # f(c) > f(d) to find the maximum
			b = d
		else:
			a = c
	return (b + a) / 2

def plotCDFvstheory(listp1,listp2,parameters,fit=False,theory=True,neutral=True,a=-1,b=1):
	"""Plots the cumulative distribution function.
	Parameters:
	-----------
	listp1: the list of allele frequencies at time t=0
	listp2: the list of allele frequencies at time t=T
	s:     the strength of selection (MUST BE >=0)
	theta: the mutation rates
	fit:   if True, will display a modified beta with selection coefficient estimated from listp
	theory: if True, will compute a theoretical selection coefficient and display the associated
		beta distribution.
	neutral: if True, displays the neutral equilibrium distribution expected
	T: the time of listp2
	a,b: see gss"""
	theta,s,N,L,T = parameters
	fig = plt.figure()
	ax = plt.axes()
	listx,listy = getCDF(listp1)
	ax.plot(listx,listy,color="orange")
	x = np.linspace(1/(2*N),1-1/(2*N),N)
	if neutral:
		pi = x**(2*theta[0]-1) * (1-x)**(2*theta[1]-1)
		pi = pi/np.sum(pi)
		ax.plot(x,np.cumsum(pi),color="red")
	listx,listy = getCDF(listp2)
	ax.plot(listx,listy,color="cyan")
	if fit:
		Z = 2*np.mean(listp2)-1
		print("Empirical Z: "+str(Z))
		pi = x**(2*theta[0]-1) * (1-x)**(2*theta[1]-1)* np.exp(-4*s*Z*x)
		pi = pi/np.sum(pi)
		ax.plot(x,np.cumsum(pi),color="blue")
	if theory:
		Z = gss(lambda Z:sqZEZ(Z,s,theta,N),a=a,b=b)
		print("Theoretical Z: "+str(Z))
		pi = x**(2*theta[0]-1) * (1-x)**(2*theta[1]-1)* np.exp(-4*s*Z*x)
		pi = pi/np.sum(pi)
		ax.plot(x,np.cumsum(pi),color="blue")
	ax.legend(["Empirical CDF (T=0)",
			"Theoretical CDF (T=0)",
			"Empirical CDF (T="+str(T)+")",
			"Theoretical CDF (T="+str(T)+")"])
	ax.set_title(
"Empirical vs Theoretical cumulative distribution function \n of allele frequencies (N="+str(N)+",L="+str(L)+",s="+str(s)+",θ="+str(theta)+",T="+str(T)+")"
		)
	plt.show()
