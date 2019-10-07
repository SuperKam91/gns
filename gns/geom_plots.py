#import standard modules 
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

#import custom modules

def plotCircle(r = 1.):
	"""
	Plot circle with radius r
	"""
	phi = np.linspace(0., 2. * np.pi, 100)
	x = r * cos(phi)
	y = r * sin(phi)
	fig = plt.figure()
	ax = fig.add_subplot(1,1)
	ax.scatter(x, y)
	plt.show()

def plotTorus(R = 1., r = 1.):
	"""
	Plot torus with major radius (distance from centre of hole) R
	and minor radius (distance from centre of tube of torus) r
	n.b. phi is azimuthal angle (angle from centre of hole),
	theta is angle from centre of tube
	"""
	theta = np.linspace(0, 2. * np.pi, 100) 
	phi = np.linspace(0, 2. * np.pi, 100)
	theta, phi = np.meshgrid(theta, phi)
	x = (R + r * np.cos(theta)) * np.cos(phi)
	y = (R + r * np.cos(theta)) * np.sin(phi)
	z = r * np.sin(theta)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection='3d')
	ax.scatter(xs = x, ys = y, zs = z, zdir = 'z', alpha=0.1)
	plt.show()

def plotSphere(r = 1.):
	"""
	Plot surface of a sphere with radius r
	"""
	theta = np.linspace(0, np.pi, 100) 
	phi = np.linspace(0, 2. * np.pi, 100)
	theta, phi = np.meshgrid(theta, phi)
	x = r * np.cos(phi) * np.sin(theta)
	y = r * np.sin(phi) * np.sin(theta)
	z = r * np.cos(theta)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection='3d')
	ax.scatter(xs = x, ys = y, zs = z, zdir = 'z', alpha=0.1)
	plt.show()