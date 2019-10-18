#import standard modules 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import sys
try:
	import cartopy.crs
except:
	pass
import matplotlib.gridspec

#import custom modules
from . import input
try:
	import spherical_kde
except:
	pass

############plotting functions

def plotLhood(x, Lhood, space):
	if space == 'log':
		Lhood = np.exp(Lhood)
	plt.figure('Lhood versus param')
	plt.scatter(x, Lhood)
	plt.show()
	plt.close()

def plotPhysPosteriorIW(x, unnormalisedSamples, Z, space):
	"""
	Plots posterior in physical space according to importance weights w(theta)L(theta) / Z. Doesn't use KDE so isn't true shape of posterior. 
	If inputting logWeights/ logZ then set space == 'log'
	"""
	if space == 'log':
		normalisedSamples = np.exp(unnormalisedSamples - Z)
	else:
		normalisedSamples = unnormalisedSamples / Z
	plt.figure('phys posterior')
	plt.scatter(x, normalisedSamples)
	plt.show()  
	plt.close()

def plotXPosterior(X, L, Z, space):
	"""
	Plots X*L(X)/Z in log X space, not including KDE methods
	"""
	if space == 'log':
		LhoodDivZ = np.exp(L - Z)
		X = np.exp(X)
	else:
		LhoodDivZ = L / Z
	LXovrZ = X * LhoodDivZ
	plt.figure('posterior')
	plt.scatter(X, LXovrZ)
	plt.set_xscale('log')
	plt.show()
	plt.close()

def callGetDist(chainsFilePrefix, plotName, nParams, plotLegend):
	"""
	produces triangular posterior plots using getDist for first nParams
	parameters from chains file as labelled in that file and in .paramnames
	plotName should contain image type extension (e.g. .png)
	"""
	print(plotName)
	try:
		import getdist.plots, getdist.loadMCSamples
	except ImportError:
		try:
			import getdist
		except ImportError:	
			print("can't import getdist. Exiting...")
			sys.exit(1)
	save = True
	paramList = ['p' + str(i+1) for i in range(nParams)]
	chains = [getdist.loadMCSamples(chain) for chain in chainsFilePrefix]
	g = getdist.plots.getSubplotPlotter(width_inch = 6)
	g.triangle_plot(chains, paramList,
	filled = False, legend_labels = plotLegend)
	if save:
		g.export(plotName)
	else:
		plt.show()
	return g

def GetDistPlotterTheor(g, p, x, plotName, Z = 1.):
	"""
	Takes GetDistPlotter object and plots theoretical posterior (p) on its axis
	for given x values. Currently only works in 1-d. 
	Optionally normalises points w.r.t. Z
	"""
	ax=g.subplots[0,0]
	ax.plot(x, p / Z, 'k')
	g.export(plotName)

def get2DMarg(params, Lhood, n):
	"""
	marginalise Lhood for 2d array params,
	with n samples in each dimension (n^2 total).
	Returns n sized array of Lhoods and n sized array of params
	for each dimension
	"""
	p1 = params[:n,0]
	L1 = np.zeros(n)
	p2 = params[::n, 1]
	L2 = np.zeros(n)
	for j in range(n):
		L1[j] = Lhood[j::n].sum()
		L2[j] = Lhood[j * n: (j+1) * n].sum()
	return p1, L1, p2, L2
	
def get3DMarg(params, Lhood, n):
	"""
	marginalise Lhood for 3d array params,
	with n samples in each dimension (n^3 total).
	Returns n sized array of Lhoods and n sized array of params
	for each dimension
	"""
	p3 = params[:n, 2]
	L3 = np.zeros(n)
	p2 = params[:n * n * n:n * n, 1]
	L2 = np.zeros(n)
	p1 = params[:n * n:n, 0]
	L1 = np.zeros(n)
	for j in range(n):
		L3[j] = Lhood[j::n].sum()
		L2[j] = Lhood[j * n * n:(j + 1) * n * n].sum()
		L1Indices = np.array([list(range(i, i + n)) for i in range(j * n, n * n * n, n * n)]) # for each j, gives a n x n shaped array
		L1[j] = Lhood[L1Indices].sum()
	return p1, L1, p2, L2, p3, L3


def get4DMarg(params, Lhood, n):
	"""
	marginalise Lhood for 4d array params,
	with n samples in each dimension (n^4 total).
	Returns n sized array of Lhoods and n sized array of params
	for each dimension
	"""
	p4 = params[:n, 3]
	L4 = np.zeros(n)
	p3 = params[:n * n:n, 2]
	L3 = np.zeros(n)
	p2 = params[::n * n * n, 1]
	L2 = np.zeros(n)
	p1 = params[:n * n * n:n * n, 0]
	L1 = np.zeros(n)
	for j in range(n):
		L4[j] = Lhood[j::n].sum()
		L3Indices = np.array([list(range(i, i + n)) for i in range(j * n, n * n * n * n, n * n)]) # for each j, gives a n^2 x n shaped array
		L3[j] = Lhood[L3Indices].sum()
		L2[j] = Lhood[j * n * n * n:(j + 1) * n * n * n].sum()
		L1Indices = np.array([list(range(i, i + n * n)) for i in range(j * n * n, n * n * n * n, n * n * n)]) # for each j, gives a n x n^2 shaped array
		L1[j] = Lhood[L1Indices].sum()
	return p1, L1, p2, L2, p3, L3, p4, L4

def get5DMarg(params, Lhood, n):
	"""
	marginalise Lhood for 5d array params,
	with n samples in each dimension (n^5 total).
	Returns n sized array of Lhoods and n sized array of params
	for each dimension
	"""
	p5 = params[:n, 4]
	L5 = np.zeros(n)
	p4 = params[:n * n:n, 3]
	L4 = np.zeros(n)
	p3 = params[:n * n * n:n * n, 2]
	L3 = np.zeros(n)
	p2 = params[::n * n * n * n, 1]
	L2 = np.zeros(n)
	p1 = params[:n * n * n * n:n * n * n, 0]
	L1 = np.zeros(n)
	for j in range(n):
		L5[j] = Lhood[j::n].sum()
		L4Indices = np.array([list(range(i, i + n)) for i in range(j * n, n * n * n * n * n, n * n)]) # for each j, gives a n^3 x n shaped array
		L4[j] = Lhood[L4Indices].sum() 
		L3Indices = np.array([list(range(i, i + n * n)) for i in range(j * n * n, n * n * n * n * n, n * n * n)]) # for each j, gives a n^2 x n^2 shaped array
		L3[j] = Lhood[L3Indices].sum()
		L2[j] = Lhood[j * n * n * n * n:(j + 1) * n * n * n * n].sum()
		L1Indices = np.array([list(range(i, i + n * n * n)) for i in range(j * n * n * n, n * n * n * n * n, n * n * n * n)]) # for each j, gives a n x n^3 shaped array
		L1[j] = Lhood[L1Indices].sum()
	return p1, L1, p2, L2, p3, L3, p4, L4, p5, L5

def cornerPlots(chainsFilePrefix, plotName, plotLegend, labels):
	try:
		import corner
	except ImportError:
		print("can't import corner. Exiting...")
		sys.exit(1)
	colours = ['black', 'blue', 'red', 'green']
	patchesList = []
	levels = [0.39346934, 0.86466472]
	#levels = None
	save = True
	for i, f in enumerate(chainsFilePrefix):
		patchesList.append(matplotlib.patches.Patch(color=colours[i])) #required to manually insert legend for histograms
		weights, _, params = input.getFromTxt(f + '.txt')
		try:
			figure = corner.corner(xs = params, weights = weights, labels=labels, color = colours[i], plot_density = False, plot_datapoints = False, levels = levels, fig = figure)
		except NameError:
			figure = corner.corner(xs = params, weights = weights, labels=labels, color = colours[i], plot_density = False, plot_datapoints = False, levels = levels)
	#figure.legend(patchesList, plotLegend) 
	if save:
		plt.savefig(plotName)
	else:
		plt.show()
	plt.close()

#spherical KDE plotting

def plotSphericalKDE(chains1, chains2 = None):
	"""
	Largely copied from main.py in spherical_kde package Will made,
	but edited to work for my chains (for gns and mn)
	chains strings should include '.txt'
	if chains2 isn't provided, uses 3rd and 4th parameters of chains1 for second plot
	"""
	# Set up a grid of figures
	fig = plt.figure(figsize=(12, 12))
	gs_vert = matplotlib.gridspec.GridSpec(3, 1)
	gs_up = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vert[0])
	gs_mid = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_vert[1])
	gs_down = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_vert[2])
	fig.add_subplot(gs_up[0], projection=cartopy.crs.Mollweide())
	fig.add_subplot(gs_up[1], projection=cartopy.crs.Mollweide())
	fig.add_subplot(gs_mid[0], projection=cartopy.crs.Orthographic())
	fig.add_subplot(gs_mid[1], projection=cartopy.crs.Orthographic(0, 90))
	fig.add_subplot(gs_mid[2], projection=cartopy.crs.Orthographic())
	fig.add_subplot(gs_mid[3], projection=cartopy.crs.Orthographic(0,90))
	fig.add_subplot(gs_down[0], projection=cartopy.crs.PlateCarree())
	fig.add_subplot(gs_down[1], projection=cartopy.crs.PlateCarree())
	weights1, _, params1 = input.getFromTxt(chains1)
	KDE1 = spherical_kde.SphericalKDE(params1[:,0], params1[:,1], weights = weights1)
	try:
		weights2, _, params2 = input.getFromTxt(chains2)
		KDE2 = spherical_kde.SphericalKDE(params2[:,0], params2[:,1], weights = weights2)
	except TypeError:
		KDE2 = spherical_kde.SphericalKDE(params1[:,2], params1[:,3], weights = weights1)		
	#chains1 plots
	fig.axes[0].gridlines()
	KDE1.plot(fig.axes[0], 'g')
	fig.axes[2].gridlines()
	KDE1.plot(fig.axes[2], 'g')
	fig.axes[3].gridlines()
	KDE1.plot(fig.axes[3], 'g')
	fig.axes[6].gridlines()
	KDE1.plot(fig.axes[6], 'g')
	#chains1 samples
	#[KDE1.plot_samples(ax) for ax in [fig.axes[i] for i in [0, 2, 3, 6]]]
	#chains2 plots
	fig.axes[1].gridlines()
	KDE2.plot(fig.axes[1], 'r')
	fig.axes[4].gridlines()
	KDE2.plot(fig.axes[4], 'r')
	fig.axes[5].gridlines()
	KDE2.plot(fig.axes[5], 'r')
	fig.axes[7].gridlines()
	KDE2.plot(fig.axes[7], 'r')
	#chains2 samples
	#[KDE2.plot_samples(ax) for ax in [fig.axes[i] for i in [1, 4, 5, 7]]]

	plt.show()
