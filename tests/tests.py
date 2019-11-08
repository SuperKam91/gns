#import standard modules 
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import time

run_repo = True
#import custom modules
if run_repo: #run gns code from repo which this test file sits in
	import context
else: #run installed package
	pass
import gns.nested_run as nested_run
import gns.plotting as plotting
import gns.toy_models as toy_models
import gns.array_checks as array_checks
import gns.theoretical_funcs as theoretical_funcs
import gns.output as output
import gns.input as input

TOL = 1e-6

def assert_same(x, y, tol):
	"""
	Asserts the floats x and y are within tol of each other
	i.e. |x - y| < tol. Returns boolean according to this,
	and |x - y| itself
	"""
	diff = np.abs(x - y)
	return diff < tol, diff

def test_gns_stats_circle():
	"""
	Test correct stats are generated for geometric nested sampler for circle toy model. 
	Note these values have been compared with the theoretically correct values.
	"""
	ElnZTrue = -1.778287448059121
	VarlnZTrue = 0.00222452082689939
	HTrue = 1.009856152283123
	paramGeomLists = [['wrapped']] #wrap parameter geometric nested sampler for von Mises distribution (circle)
	doPlots = [False]
	return_vals = True
	shapeList = ['circle'] #run circle toy model, simple von Mises distribution
	shapeSuffixes = ['_Sc']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	ElnZPred, VarlnZPred, HPred = runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots, return_vals = True)
	bool_same, diff = assert_same(ElnZTrue, ElnZPred, TOL)
	if bool_same:
		print("E[ln(Z)] test for circle model complete.")
	else:
		print("E[ln(Z)] test for circle model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(VarlnZTrue, VarlnZPred, TOL)
	if bool_same:
		print("Var[ln(Z)] test for circle model complete.")
	else:
		print("Var[ln(Z)] test for circle model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(HTrue, HPred, TOL)
	if bool_same:
		print("K-L divergence test for circle model complete.")
	else:
		print("K-L divergence test for circle model failed. Discrepancy between true and obtained value is " + str(diff))

def test_gns_stats_torus():
	"""
	Test correct stats are generated for geometric nested sampler for torus toy model. 
	Note these values have been compared with the theoretically correct values.
	"""
	ElnZTrue = -3.5870812483323076
	VarlnZTrue = 0.004308045833100849
	HTrue = 2.013937631676229
	paramGeomLists = [['wrapped', 'wrapped']] #wrap parameter geometric nested sampler for torus
	doPlots = [True]
	shapeList = ['torus'] #run torus toy model, simple von Mises distribution
	shapeSuffixes = ['_St']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	ElnZPred, VarlnZPred, HPred = runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots, return_vals = True)
	bool_same, diff = assert_same(ElnZTrue, ElnZPred, TOL)
	if bool_same:
		print("E[ln(Z)] test for torus model complete.")
	else:
		print("E[ln(Z)] test for torus model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(VarlnZTrue, VarlnZPred, TOL)
	if bool_same:
		print("Var[ln(Z)] test for torus model complete.")
	else:
		print("Var[ln(Z)] test for torus model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(HTrue, HPred, TOL)
	if bool_same:
		print("K-L divergence test for torus model complete.")
	else:
		print("K-L divergence test for torus model failed. Discrepancy between true and obtained value is " + str(diff))

def test_gns_stats_sphere():
	"""
	Test correct stats are generated for geometric nested sampler for sphere toy model. 
	Note these values have been compared with the theoretically correct values.
	"""
	ElnZTrue = -3.024004031644396
	VarlnZTrue = 0.0033782554288421807
	HTrue = 1.549216368953933
	paramGeomLists = [['sphere', 'sphere']] #spherical parameters for sphere
	doPlots = [True]
	shapeList = ['sphere'] #run sphere toy model, simple von Mises distribution
	shapeSuffixes = ['_Ss']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	ElnZPred, VarlnZPred, HPred = runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots, return_vals = True)
	bool_same, diff = assert_same(ElnZTrue, ElnZPred, TOL)
	if bool_same:
		print("E[ln(Z)] test for sphere model complete.")
	else:
		print("E[ln(Z)] test for sphere model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(VarlnZTrue, VarlnZPred, TOL)
	if bool_same:
		print("Var[ln(Z)] test for sphere model complete.")
	else:
		print("Var[ln(Z)] test for sphere model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(HTrue, HPred, TOL)
	if bool_same:
		print("K-L divergence test for sphere model complete.")
	else:
		print("K-L divergence test for sphere model failed. Discrepancy between true and obtained value is " + str(diff))

def test_gns_stats_6sphereVIII():
	"""
	Test correct stats are generated for geometric nested sampler for 6-sphere VIII toy model. 
	WARNING this test takes a while to run (~5 hours)
	"""	
	ElnZTrue = 2102.1710800835644
	VarlnZTrue = 0.0504359918722912
	HTrue = 25.255793511215728
	paramGeomLists = [['sphere', 'sphere']*6] #spherical parameters geometric nested sampler for 6-sphere
	doPlots = [True]
	shapeList = ['6 sphere VIII'] #run 6-sphete VIII toy model, kent distributions defined on 6 separate spheres
	shapeSuffixes = ['_St']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	ElnZPred, VarlnZPred, HPred = runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots, return_vals = True)
	bool_same, diff = assert_same(ElnZTrue, ElnZPred, TOL)
	if bool_same:
		print("E[ln(Z)] test for 6-sphere VIII model complete.")
	else:
		print("E[ln(Z)] test for 6-sphere VIII model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(VarlnZTrue, VarlnZPred, TOL)
	if bool_same:
		print("Var[ln(Z)] test for 6-sphere VIII model complete.")
	else:
		print("Var[ln(Z)] test for 6-sphere VIII model failed. Discrepancy between true and obtained value is " + str(diff))
	bool_same, diff = assert_same(HTrue, HPred, TOL)
	if bool_same:
		print("K-L divergence test for 6-sphere VIII model complete.")
	else:
		print("K-L divergence test for 6-sphere VIII model failed. Discrepancy between true and obtained value is " + str(diff))

def test_gns_plot_circle():
	"""
	Test correct plot is generated for geometric nested sampler for circle toy model. 
	Should be compared with ../image_output/toy_models_empirical_getdist/circle.png
	"""
	paramGeomLists = [['wrapped']] #wrap parameter geometric nested sampler for von Mises distribution (circle)
	doPlots = [True]
	shapeList = ['circle'] #run circle toy model, simple von Mises distribution
	shapeSuffixes = ['_Sc']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots)

def test_gns_plot_torus():
	"""
	Test correct plot is generated for geometric nested sampler for torus toy model. 
	Should be compared with ../image_output/toy_models_empirical_getdist/circle.png
	"""
	paramGeomLists = [['wrapped', 'wrapped']] #wrap parameter geometric nested sampler for torus
	doPlots = [True]
	shapeList = ['torus'] #run torus toy model, simple von Mises distribution
	shapeSuffixes = ['_St']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots)

def test_gns_plot_sphere():
	"""
	Test correct plot is generated for geometric nested sampler for torus toy model. 
	Should be compared with ../image_output/toy_models_empirical_getdist/sphere.png
	"""
	paramGeomLists = [['sphere', 'sphere']] #spherical parameters for sphere
	doPlots = [True]
	shapeList = ['sphere'] #run sphere toy model, simple von Mises distribution
	shapeSuffixes = ['_Ss']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots)

def test_gns_plot_6sphereVIII():
	"""
	Test correct plot is generated for geometric nested sampler for 6 sphere VIII toy model (see toy_models.py). 
	Should be compared with ../image_output/toy_models_empirical_getdist/kent_6.png
	WARNING this test takes a while to run (~5 hours)
	"""
	paramGeomLists = [['sphere', 'sphere']*6] #spherical parameters geometric nested sampler for 6-sphere
	doPlots = [True]
	shapeList = ['6 sphere VIII'] #run 6-sphete VIII toy model, kent distributions defined on 6 separate spheres
	shapeSuffixes = ['_St']
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	samplerSuffixes = ['_mhwg'] 
	runTests(shapeList, shapeSuffixes, samplerList, samplerSuffixes, paramGeomLists, doPlots)

def runTests(shapeList = ['circle', 'torus', '6 sphere VIII'], 
	shapeSuffixes = ['_Sc', '_St', '_S6s8'], 
	samplerList = ['MH', 'MH WG'], 
	samplerSuffixes = ['_mh', '_mhwg'], 
	paramGeomLists = [['wrapped'], ['wrapped', 'wrapped'], ['sphere', 'sphere']*6], 
	doPlots = [True, True, True],
	return_vals = False):
	"""
	Run geometric nested sampling tests specified by function arguments.
	By default, runs MH sampler, GNS sampler for three toy models, and plots posteriors using getdist.
	These three toy models full exploit the different aspects of the GNS algorithm.

	shapeList: list of str, which toy models to run. See toy_models.py for available toy models
	shapeSuffixes: list of str, should correspond to shapeList value(s). Used to index saved files
	samplerList: list of str, list specifying which algorithms to run on each problem, 
	namely 'MH WG': geometric nested sampling, 'MH': vanilla Metropolis Hastings nested sampling, 'MN': MultiNest
	samplerSuffixes: list of str, should correspond to samplerList value(s). Used to index saved files
	paramGeomLists: list of lists of str, as described in README
	doPlots: list of bool, whether to generate getdist plots for each sampler. Should be same length as samplerList
	return_vals : bool whether to return statistics parameters of nested run or not, 
	i.e. the expected log evidence E[ln(Z)], its variance var[ln(Z)] and the K-L divergence H.
	If False, writes these values to file instead (and prints to stdout).

	"""
	#additional I/O details and configuration for toy model runs
	###########################################################################
	seed = 0
	setupDict = {'verbose':True, 'trapezoidalFlag': False, 'ZLiveType':'average X', 'terminationType':'evidence', 'terminationFactor': 0.1, 'sampler':None, 'outputFile':None, 'space':'log', 'paramGeomList':None}
	lp = 'h' #as a guide: l for low livepoints (50), i for intermediate livepoints (100), h for high livepoints (500), m for massive livepoints (2000)
	calcZ = False #calculate theoretical Z
	calcPost = False #calculate theoretical posterior 'samples'
	doRun = True
	plotSeparate = False #whether to plot different samplers on same figure or not
	theoryList = ['grid'] #how to calculate theoretical values if calcPost == True | calcZ == True
	theorySuffixes = ['_tg'] #should correspond to theoryList values. Used to index saved files
	outputFile = '../text_output/t' + lp
	plotFile = '../image_output/toy_models_empirical_getdist/t' + lp
	theorPostFile = outputFile[:-1]
	###############################################################################

	for i, shape in enumerate(shapeList):
		chainsFilesPrefixes = []
		plotLegend = []
		paramNames, priorParams, LhoodParams = toy_models.getToyHypersGeom(shape)
		nDims = len(paramNames)
		array_checks.checkInputParamsShape(priorParams, LhoodParams, nDims)
		targetSupport = toy_models.getTargetSupport(priorParams)
		priorFunc, logPriorFunc, invPriorFunc, LhoodFunc, LLhoodFunc = toy_models.getToyFuncs(priorParams, LhoodParams)
		shapeSuffix = shapeSuffixes[i]
		outputFile2 = outputFile + shapeSuffix
		theorPostFile2 = theorPostFile + shapeSuffix
		if calcZ:
			Z, ZErr = theoretical_funcs.calcZTheor(logPriorFunc, LLhoodFunc, targetSupport, nDims)
			output.writeTheorZ(Z, ZErr, theorPostFile2)
		if calcPost:
			for j, theory in enumerate(theoryList):
				if calcPost:
					output.writeTheoreticalSamples(theorPostFile2, logPriorFunc, invPriorFunc, LLhoodFunc, targetSupport, paramNames, method = theory)
				theorySuffix = theorySuffixes[j]
				chainsFilesPrefixes.append(theorPostFile2 + theorySuffix)
				plotLegend.append(theorPostFile2[16:] + theorySuffix)
		for j, sampler in enumerate(samplerList):
			if seed is not None:
				np.random.seed(seed) #set this to value if you want NS to use same randomisations
			setupDict['paramGeomList'] = paramGeomLists[i]
			samplerSuffix = samplerSuffixes[j]
			setupDict['sampler'] = sampler
			setupDict['outputFile'] = outputFile2 + samplerSuffix
			if sampler != 'MN' and sampler != 'MNW': 
				setupDict['outputFile'] = '../text_output/t' + lp + shapeSuffixes[i] + samplerSuffixes[j]
				if return_vals:
					return nested_run.NestedRun(priorFunc, invPriorFunc, LhoodFunc, paramNames, targetSupport, setupDict, LLhoodFunc, return_vals)
			else:
				#run multinest
				import gns.mn_run as mn_run
				if sampler == 'MN':
					wrapped = False
				elif sampler == 'MNW':
					wrapped = True
				mn_run.MNRun2(invPriorFunc, LLhoodFunc, paramNames, setupDict['outputFile'], wrapped)
				output.writeRanges(setupDict['outputFile'], paramNames, targetSupport)
				output.writeParamNames(setupDict['outputFile'], paramNames)
			chainsFilesPrefixes.append(setupDict['outputFile'])
			plotLegend.append(setupDict['outputFile'][17:])
		#do plots
		if np.any(doPlots):
			ext = samplerSuffixes
			chainsFilesPrefixes = list(itertools.compress(chainsFilesPrefixes, doPlots))
			plotLegend = list(itertools.compress(plotLegend, doPlots))
			np.seterr(all = 'ignore')
			if plotSeparate:
				for j, chains in enumerate(chainsFilesPrefixes):
					plotting.callGetDist([chains], plotFile + shapeSuffix + ext[j] + '.png', nDims, [plotLegend[j]])
			else:
				if len(ext) < 4: #not plotting all samplers and theor
					sampExt = ''.join(ext)
				else:
					sampExt = ''
				plotting.callGetDist(chainsFilesPrefixes, plotFile + shapeSuffix + sampExt + '.png', nDims, plotLegend)

def singleRun():
	"""
	Variables are as described in runTests()
	"""
	calcZ = False #Calculate theoretical evidence value (not recommended for high dimensions)
	calcPost = False #Calculate theoretical posterior (not recommended for high dimensions)
	doRun = True
	doPlot = True  #corresponds to sampler samples
	shape = 'sphere II' #'circle', 'torus', 'sphere'
	shapeSuffix = '_Ss' #'_Sc', '_St', '_Ss'
	theory = 'grid' #'grid', 'samples'
	theorySuffixes = '_tg' #'_tg', '_ts'
	sampler = 'MH WG' # 'blind', 'MH', 'MH WG', 'MN'
	samplerSuffix = '_mhwg' # '_b', '_mh', '_mhwg', '_mn'
	outputFile = '../text_output/t'
	plotFile = '../image_output/t'
	setupDict = {'verbose': True, 'trapezoidalFlag': False, 'ZLiveType':'average X', 'terminationType':'evidence', 'terminationFactor': 0.01, 'sampler': None, 'outputFile': None, 'space':'linear', 'paramGeomList':None}
	paramNames, priorParams, LhoodParams = toy_models.getToyHypersGeom(shape)
	nDims = len(paramNames)
	array_checks.checkInputParamsShape(priorParams, LhoodParams, nDims)
	targetSupport = toy_models.getTargetSupport(priorParams)
	priorFunc, logPriorFunc, invPriorFunc, LhoodFunc, LLhoodFunc = toy_models.getToyFuncs(priorParams, LhoodParams)
	outputFile2 = outputFile + shapeSuffix
	if calcZ:
		Z, ZErr = theoretical_funcs.calcZTheor(logPriorFunc, LLhoodFunc, targetSupport, nDims)
		output.writeTheorZ(Z, ZErr, outputFile2)
	if calcPost:
		output.writeTheoreticalSamples(outputFile2, logPriorFunc, invPriorFunc, LLhoodFunc, targetSupport, paramNames, method = theory)
	np.random.seed(0) #set this to value if you want NS to use same randomisations
	setupDict['sampler'] = sampler
	setupDict['outputFile'] = outputFile2 + samplerSuffix
	if doRun:
		if sampler == 'MH WG':
			paramGeomList = ['sphere', 'sphere'] # ['wrapped'], ['wrapped', 'wrapped'], ['sphere', 'sphere']
		else:
			paramGeomList = ['vanilla', 'vanilla'] # ['vanilla'], ['vanilla', 'vanilla'], ['vanilla', 'vanilla']
		setupDict['paramGeomList'] = paramGeomList
		if sampler != 'MN': 
			nested_run.NestedRun(priorFunc, invPriorFunc, LhoodFunc, paramNames, targetSupport, setupDict, LLhoodFunc)
		else:
			import mn_run
			mn_run.MNRun2(invPriorFunc, LLhoodFunc, paramNames, setupDict['outputFile'])
			output.writeRanges(setupDict['outputFile'], paramNames, targetSupport)
			output.writeParamNames(setupDict['outputFile'], paramNames)
	if doPlot:
		plotting.callGetDist([setupDict['outputFile']], plotFile + shapeSuffix + '.png', nDims, setupDict['outputFile'][19:])

def shapeChainPlots(shapeSuffix, inputFile, outputFile = None, plotWeights = False):
	"""
	Plot Lhood obtained from chains files either for theoretical
	or empirical samples. For former, plots 1-d graphs of raw samples (i.e. letting all parameters)
	and also 1-d marginalised graphs. In latter case, the two are equivalent assuming no binning
	in the marginalisation
	if plotWeights, plots the weights L omega / Z instead of Lhood for empirical samples
	n.b. for mn samples, 2nd column doesn't appear to be LLhood, in fact I have no idea what it is,
	hence plotting it is pretty pointless

	Args:

	shapeSuffix : string shape suffix string of name
	inputFile : string input file location 
	outputFile : string output file location 
	plotWeights : boolean see body of docstring
	"""
	if 'tg' in inputFile or plotWeights:
		Lhood, _, params = input.getFromTxt(inputFile)
	else:
		_, LLhood, params = input.getFromTxt(inputFile)
		Lhood = np.exp(LLhood - scipy.misc.logsumexp(LLhood))
	if shapeSuffix == '_Sc':
		plt.figure('Sc')
		plt.scatter(params, Lhood)
		plt.xlabel('phi')
		if outputFile:
			plt.savefig(outputFile + '_phi.png')
		else:
			plt.show()
		plt.close()
	elif shapeSuffix == '_St':
		#unmarginalised plots
		plt.figure('St phi')
		plt.scatter(params[:,0], Lhood)
		plt.xlabel('phi')
		if outputFile:
			plt.savefig(outputFile + '_phi.png')
		else:
			plt.show()
		plt.close()
		plt.figure('St theta')
		plt.scatter(params[:,1], Lhood)
		plt.xlabel('theta')
		if outputFile:
			plt.savefig(outputFile + '_theta.png')
		else:
			plt.show()
		plt.close()
		if 'tg' in inputFile:
			#marginalised plots  
			oneDn = 100 #this has to be same as in plotTheoreticalSamples() or plots will be completely wrong!!
			phi, LPhi, theta, LTheta = plotting.get2DMarg(params, Lhood, oneDn) 
			plt.figure('St phi marg')
			plt.scatter(phi, LPhi)
			plt.xlabel('phi marg')
			if outputFile:
				plt.savefig(outputFile + '_phimarg.png')
			else:
				plt.show()
			plt.close()
			plt.figure('St theta marg')
			plt.scatter(theta, LTheta)
			plt.xlabel('theta marg')
			if outputFile:
				plt.savefig(outputFile + '_thetamarg.png')
			else:
				plt.show()
			plt.close()
	elif shapeSuffix == '_Ss':
		#unmarginalised plots
		plt.figure('Ss phi')
		plt.scatter(params[:,0], Lhood)
		plt.xlabel('phi')
		if outputFile:
			plt.savefig(outputFile + '_phi.png')
		else:
			plt.show()
		plt.close()
		plt.figure('Ss theta')
		plt.scatter(params[:,1], Lhood)
		plt.xlabel('theta')
		if outputFile:
			plt.savefig(outputFile + '_theta.png')
		else:
			plt.show()
		plt.close()
		if 'tg' in inputFile:
			#marginalised plots
			oneDn = 100 #this has to be same as in plotTheoreticalSamples() or plots will be completely wrong!!
			phi, LPhi, theta, LTheta = plotting.get2DMarg(params, Lhood, oneDn)
			plt.figure('Ss phi marg')
			plt.scatter(phi, LPhi)
			plt.xlabel('phi marg')
			if outputFile:
				plt.savefig(outputFile + '_phimarg.png')
			else:
				plt.show()
			plt.close()
			plt.figure('Ss theta marg')
			plt.scatter(theta, LTheta)
			plt.xlabel('theta marg')
			if outputFile:
				plt.savefig(outputFile + '_thetamarg.png')
			else:
				plt.show()
			plt.close()

test_gns_stats_circle()
print('Starting next test in 10 seconds...')
time.sleep(10)
test_gns_stats_torus()
print('Starting next test in 10 seconds...')
time.sleep(10)
test_gns_stats_sphere()