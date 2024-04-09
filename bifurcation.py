import matplotlib.pyplot as plt
import numpy as np
#This file draws the bifurcation diagram for stabilising/disruptive selection.

theta = (.6,.9) #mutation rates (forward, backwards)
s = 2 #strength of selection


def match_equation(Z,s,theta,epsilon):
	"""Checks if a given Z value satisfies the nonlinear equation"""
	x = np.linspace(1e-3,1-1e-3,1000)
	pi = x**(2*theta[0]-1)*(1-x)**(2*theta[1]-1)*np.exp(-s*Z*x)
	pi = pi/np.sum(pi)
	if np.abs(np.sum((2*x-1)*pi) - Z) <= epsilon:
		return True
	return False

def find_solutions(theta,epsilon=1e-3):
	list_s = np.linspace(-15,1,200)
	list_Z = np.linspace(-1,1,1001)
	list_points = []
	for s in list_s:
		for Z in list_Z:
			if match_equation(Z,s,theta,epsilon):
				list_points.append([s,Z])
	return np.transpose(np.array(list_points))

def plot_bifurcations():
	ax = plt.axes()
	for theta in [(1,1,"red"),(.5,1,"green"),(2,1,"blue")]:
		points = find_solutions(theta[:-1])
		with open("test"+str(theta)+".ndy","bw") as f:
			np.save(f,points)
		ax.plot(points[0],points[1],"o",color=theta[-1])
		print("Done: "+theta[-1])
	ax.set_ylim((-1,1))
	ax.set_xlabel("s")
	ax.set_ylabel("Z")
	ax.set_title(
"Equilibrium value of the mean phenotype for stabilising/disruptive selection")
	ax.legend(["θ1=1, θ2=1",
		"θ1=.025, θ2=.05",
		"θ1=2, θ2=1"])
	plt.show()
