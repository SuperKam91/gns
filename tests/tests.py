#import standard modules 
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt

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

def runTests():
	#configuration for toy model runs
	###########################################################################
	setupDict = {'verbose':True, 'trapezoidalFlag': False, 'ZLiveType':'average X', 'terminationType':'evidence', 'terminationFactor': 0.1, 'sampler':None, 'outputFile':None, 'space':'log', 'paramGeomList':None}
	# lp = 'l' #low livepoints (50)
	#lp = 'i' #intermediate livepoints (100)
	lp = 'h' #high livepoints (500)
	#lp = 'm' #massive livepoints (2000)
	calcZ = False #calculate theoretical Z
	calcPost = False #calculate theoretical posterior 'samples'
	doRun = True
	# doPlots = [True, True] #if more than one sampler is considered, this should be a boolean list corresponding to which samplers should be plotted, and should be same length as samplerList. Just set to False for no plottingt set to F
	doPlots = [True]
	plotter = 'getdist'
	plotSeparate = False #whether to plot different samplers on same figure or not
	# shapeList = ['circle', 'torus', '6 sphere VIII'] #run these three toy models
	shapeList = ['circle'] #run circle toy model, simple von Mises distribution
	# shapeSuffixes = ['_Sc', '_St', '_S6s8'] #should correspond to shapeList value(s). Used to index saved files
	shapeSuffixes = ['_Sc']
	theoryList = ['grid'] #how to calculate theoretical values if calcPost == True | calcZ == True
	theorySuffixes = ['_tg'] #should correspond to theoryList values. Used to index saved files
	# samplerList = ['MH', 'MH WG'] #run both geometric MH nested sampling and vanilla MH nested sampling, 
	samplerList = ['MH WG'] #just do geometric MH nested sampling
	# samplerSuffixes = ['_mh', '_mhwg']
	samplerSuffixes = ['_mhwg'] #should correspond to samplerList value(s). Used to index saved files
	outputFile = '../text_output/t' + lp
	plotFile = '../image_output/toy_models_empirical_getdist/t' + lp
	# plotFile = '../image_output/toy_models_empirical_corner/t' + lp
	theorPostFile = outputFile[:-1]
	shapesLabelsList = [[r"$\phi$"], [r"$\phi$", r"$\theta$"], ['$\\phi_{1}', '$\\theta_{1}', '$\\phi_{2}$', '$\\theta_{2}$', '$\\phi_{3}$', '$\\theta_{3}$', '$\\phi_{4}$', '$\\theta_{4}$', '$\\phi_{5}$', '$\\theta_{5}$', '$\\phi_{6}$', '$\\theta_{6}$']] #parameter names for plots. Should be same length as shapeList
	# shapesLabelsList = [[r"$\phi$"]] #parameter names
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
		shapesLabels = shapesLabelsList[i]
		if calcZ:
			Z, ZErr = theoretical_funcs.calcZTheor(logPriorFunc, LLhoodFunc, targetSupport, nDims)
			#np.exp(theoretical_funcs.integrateLogFunc(logPriorFunc, LLhoodFunc, targetSupport)) #my rubbish log integrator
			output.writeTheorZ(Z, ZErr, theorPostFile2)
		if calcPost:
			for j, theory in enumerate(theoryList):
				if calcPost:
					output.writeTheoreticalSamples(theorPostFile2, logPriorFunc, invPriorFunc, LLhoodFunc, targetSupport, paramNames, method = theory)
				theorySuffix = theorySuffixes[j]
				chainsFilesPrefixes.append(theorPostFile2 + theorySuffix)
				plotLegend.append(theorPostFile2[16:] + theorySuffix)
		for j, sampler in enumerate(samplerList):
			np.random.seed(0) #set this to value if you want NS to use same randomisations
			samplerSuffix = samplerSuffixes[j]
			setupDict['sampler'] = sampler
			setupDict['outputFile'] = outputFile2 + samplerSuffix
			if doRun:
				if sampler == 'MH WG':
					#what modes to calculate different parameters in for geometric nested sampler. either 'vanilla', 'wrapped' or 'sphere'. length of outer list
					#should be same length as shapeList. Length of each inner list should be same as number of parameters for that model (see toy_models.py)
					# paramGeomLists = [['wrapped'], ['wrapped', 'wrapped'], ['sphere', 'sphere']*6] 
					paramGeomLists = [['wrapped']] #wrap parameter geometric nested sampler for von Mises distribution (circle)
				else:
					#for vanilla MH sampler, all modes must be vanilla
					# paramGeomLists = [['vanilla'], ['vanilla', 'vanilla'], ['vanilla', 'vanilla']*6]
					paramGeomLists = [['vanilla']]
				setupDict['paramGeomList'] = paramGeomLists[i]
				if sampler != 'MN' and sampler != 'MNW': 
					setupDict['outputFile'] = '../text_output/t' + lp + shapeSuffixes[i] + samplerSuffixes[j]
					nested_run.NestedRun(priorFunc, invPriorFunc, LhoodFunc, paramNames, targetSupport, setupDict, LLhoodFunc)
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
			if 'getdist' in plotter:
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
		#np.exp(theoretical_funcs.integrateLogFunc(logPriorFunc, LLhoodFunc, targetSupport)) #my rubbish log integrator
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

runTests()