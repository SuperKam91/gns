#import standard modules 
import numpy as np
import sys

#import custom modules
import ns_end_funcs
import ns_loop_funcs
import prob_funcs
import samplers
import calculations
import recurrence_calculations
import keeton_calculations
import theoretical_funcs
import output
import array_checks
import geom_sampler

############nested run functions

def NestedRun(priorFunc, invPriorFunc, LhoodFunc, paramNames, targetSupport, setupDict, LLhoodFunc = None):
	"""
	Wrapper around linear and log nested run functions.
	"""
	if setupDict['space'] == 'linear':
		NestedRunLinear(priorFunc, invPriorFunc, LhoodFunc, paramNames, targetSupport, setupDict)
	elif setupDict['space'] == 'log':
		NestedRunLog(priorFunc, invPriorFunc, LLhoodFunc, paramNames, targetSupport, setupDict)

def NestedRunLog(priorFunc, invPriorFunc, LLhoodFunc, paramNames, targetSupport, setupDict):
	"""
	function which completes a NS run. parameters of priors and likelihood need to be specified, as well as a flag indication type of prior for each dimension and the pdf for the lhood.
	setupDict contains other setup parameters such as termination type & factor, method of finding new livepoint, details of how weights are calculated, how final Z contribution is added, and directory/file prefix for saved files.
	"""
	nLive = 500 #low value is 50, high value is 500
	nDims = len(paramNames)
	array_checks.checkTargSupShape(targetSupport, nDims)
	livePoints = np.random.rand(nLive, nDims) #initialise livepoints to random values uniformly on [0,1]^D
	livePointsPhys = invPriorFunc(livePoints) #Convert livepoint values to physical values	
	array_checks.checkinvPriorShape(livePointsPhys, livePoints.shape)
	livePointsLLhood = LLhoodFunc(livePointsPhys) #calculate LLhood values of initial livepoints
	livePointsLLhood = array_checks.checkLhoodShape(livePointsLLhood, nLive)
	# initialise lists for storing values
	logEofXArr = []
	logEofWeights = []
	deadPoints = []
	deadPointsPhys = []
	deadPointsLLhood = []	
	# initialise mean and variance of Z variables and other moments
	logEofZ = -np.inf
	logEofZ2 = -np.inf
	logEofZX = -np.inf
	logEofX = 0.
	logEofX2 = 0.
	logX = 0.
	# initialise other variables
	LLhoodStar = -np.inf
	H = 0.
	nest = 0
	logZLive = np.inf
	checkTermination = 100
	#if 'geom' in setupDict['sampler']: #temporary whilst testing geometric ns
	nonGeomList, boundaryList, geomList, shapeList = geom_sampler.splitGeomParams(setupDict['paramGeomList']) 
	circleList, torusList, sphereList = geom_sampler.splitGeomShapes(geomList, shapeList)
	nonGeomLowerLimits, nonGeomUpperLimits = geom_sampler.getNonGeomLimits(targetSupport, nonGeomList)
	circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits = geom_sampler.getShapeLimits(targetSupport, circleList, torusList, sphereList)
	#elif 'MH' in setupDict['sampler']: #temporary whilst testing geometric ns
	#	vanillaList, circleList, torusList, sphereList = [], [], [], [] #temporary whilst testing geometric ns
	#	vanillaLowerLimits, vanillaUpperLimits, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]) #temporary whilst testing geometric ns
	while True:
		LLhoodStarOld = LLhoodStar 
		deadIndex = np.argmin(livePointsLLhood) #index of lowest likelihood livepoint (next deadpoint)
		LLhoodStar = livePointsLLhood[deadIndex] #LLhood of dead point and new target
		#update expected values of moments of X and Z, and get posterior weights
		logEofZNew, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight = recurrence_calculations.updateLogZnXMoments(nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, LLhoodStarOld, LLhoodStar, setupDict['trapezoidalFlag'])
		logEofXArr.append(logEofX)
		logEofWeights.append(logEofWeight)
		H = recurrence_calculations.updateHLog(H, logEofWeight, logEofZNew, LLhoodStar, logEofZ)
		logEofZ = logEofZNew #update evidence part II
		#WARNING, VIEWING A NUMPY SLICE (IE NOT USING NP.COPY) DOES NOT CREATE A COPY AND SO A-POSTORI CHANGES TO ARRAY WILL AFFECT PREVIOUSLY SLICED ARRAY
		#USE NP.MAY_SHARE_MEMORY(A, B) TO SEE IF ARRAYS SHARE MEMORY, PYTHON 'IS' KEYWORD DOESN'T WORK  
		deadPointPhys = np.copy(livePointsPhys[deadIndex]).reshape(1,-1)
		deadPointsPhys.append(deadPointPhys)
		deadPointLLhood = LLhoodStar
		deadPointsLLhood.append(deadPointLLhood)
		#update array where last deadpoint was with new livepoint picked subject to L_new > L*
		if setupDict['sampler'] == 'blind':
			livePointsPhys[deadIndex], livePointsLLhood[deadIndex] = samplers.getNewLiveBlind(invPriorFunc, LLhoodFunc, LLhoodStar, nDims)
		#elif setupDict['sampler'] == 'MH' or setupDict['sampler'] == 'MH geom': #temporary whilst testing geometric ns
		elif 'MH' in setupDict['sampler']:
			livePointsPhys[deadIndex], livePointsLLhood[deadIndex] = samplers.getNewLiveMH(livePointsPhys, deadIndex, priorFunc, targetSupport, LLhoodFunc, LLhoodStar, nDims, nonGeomList, boundaryList, circleList, torusList, sphereList, nonGeomLowerLimits, nonGeomUpperLimits, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits)
		if setupDict['verbose']:
			output.printUpdate(nest, deadPointPhys, deadPointLLhood, logEofZ, livePointsPhys[deadIndex].reshape(1, -1), livePointsLLhood[deadIndex], 'log')
		nest += 1
		if nest % checkTermination == 0:
			breakFlag, liveMaxIndex, liveLLhoodMax, avLLhood, nFinal = ns_loop_funcs.tryTerminationLog(setupDict['verbose'], setupDict['terminationType'], setupDict['terminationFactor'], nest, nLive, logEofX, livePointsLLhood, LLhoodStar, setupDict['ZLiveType'], setupDict['trapezoidalFlag'], logEofZ, H)
			if breakFlag: #termination condition was reached
				break
	#######################
	#following code may lose precision due to having to exponentiate numbers, which needs to be done
	#as functions working in log space haven't been implemented yet
	#######################
	#EofZ = np.exp(logEofZ) #debug
	#EofZ2 = np.exp(logEofZ2) #debug
	#varZ = calculations.calcVariance(EofZ, EofZ2) #debug
	logVarZ = calculations.calcVarianceLog(logEofZ, logEofZ2)
	EofLogZ = calculations.calcEofLogZ(logEofZ, logEofZ2, 'log')
	varLogZ = calculations.calcVarLogZ(logEofZ, logEofZ2, 'log')
	logEofZK, logEofZ2K = keeton_calculations.calcZMomentsKeetonLog(np.array(deadPointsLLhood), nLive, nest)
	#EofZK, EofZ2K = keeton_calculations.calcZMomentsKeeton(np.exp(np.array(deadPointsLLhood)), nLive, nest) #debug
	#varZK = calculations.calcVariance(EofZK, EofZ2K) #debug
	logVarZK = calculations.calcVarianceLog(logEofZK, logEofZ2K)
	EofLogZK = calculations.calcEofLogZ(logEofZK, logEofZ2K, 'log')
	varLogZK = calculations.calcVarLogZ(logEofZK, logEofZ2K, 'log')
	HKL = keeton_calculations.calcHKeetonLog(logEofZK, np.array(deadPointsLLhood), nLive, nest)
	#HK = keeton_calculations.calcHKeeton(EofZK, np.exp(np.array(deadPointsLLhood)), nLive, nest) #debug
	if setupDict['verbose']:
		output.printBreak()
		#output.printZHValues(EofZ, EofZ2, varZ, H, 'linear', 'before final', 'recursive') #debug
		#output.printZHValues(EofZK, EofZ2K, varZK, HK, 'linear', 'before final', 'Keeton equations') #debug
		output.printZHValues(logEofZ, logEofZ2, logVarZ, EofLogZ, varLogZ, H, 'log', 'before final', 'recursive')
		output.printZHValues(logEofZK, logEofZ2K, logVarZK, EofLogZK, varLogZK, HKL, 'log', 'before final', 'Keeton equations')
	logEofZTotal, logEofZ2Total, H, livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr = ns_end_funcs.getFinalContributionLog(setupDict['verbose'], setupDict['ZLiveType'], setupDict['trapezoidalFlag'], nFinal, logEofZ, logEofZ2, logEofX, logEofWeights, H, livePointsPhys, livePointsLLhood, avLLhood, liveLLhoodMax, liveMaxIndex, LLhoodStar)
	totalPointsPhys, totalPointsLLhood, logEofXArr, logEofWeights = ns_end_funcs.getTotal(deadPointsPhys, livePointsPhysFinal, deadPointsLLhood, livePointsLLhoodFinal, logEofXArr, logEofXFinalArr, logEofWeights)
	#EofZTotal = np.exp(logEofZTotal) #debug
	#EofZ2Total = np.exp(logEofZ2Total) #debug
	#varZ = calculations.calcVariance(EofZTotal, EofZ2Total) #debug
	logVarZTotal = calculations.calcVarianceLog(logEofZTotal, logEofZ2Total)
	EofLogZTotal = calculations.calcEofLogZ(logEofZTotal, logEofZ2Total, 'log')
	varLogZTotal = calculations.calcVarLogZ(logEofZTotal, logEofZ2Total, 'log')
	logEofZFinalK, logEofZ2FinalK = keeton_calculations.calcZMomentsFinalKeetonLog(livePointsLLhood, nLive, nest)
	#EofZFinalK, EofZ2FinalK = keeton_calculations.calcZMomentsFinalKeeton(np.exp(livePointsLLhood), nLive, nest) #debug
	#varZFinalK = calculations.calcVariance(EofZFinalK, EofZ2FinalK) #debug
	logVarZFinalK = calculations.calcVarianceLog(logEofZFinalK, logEofZ2FinalK)
	logEofZZFinalK = keeton_calculations.calcEofZZFinalKeetonLog(np.array(deadPointsLLhood), livePointsLLhood, nLive, nest)
	#EofZZFinalK = keeton_calculations.calcEofZZFinalKeeton(np.exp(np.array(deadPointsLLhood)), np.exp(livePointsLLhood), nLive, nest) #debug
	EofLogZFinalK = calculations.calcEofLogZ(logEofZFinalK, logEofZ2FinalK, 'log')
	varLogZFinalK = calculations.calcVarLogZ(logEofZFinalK, logEofZ2FinalK, 'log')
	logVarZTotalK = keeton_calculations.getVarTotalKeetonLog(logVarZK, logVarZFinalK, logEofZK, logEofZFinalK, logEofZZFinalK)
	#varZTotalK = keeton_calculations.getVarTotalKeeton(varZK, varZFinalK, EofZK, EofZFinalK, EofZZFinalK) #debug
	logEofZTotalK = keeton_calculations.getEofZTotalKeetonLog(logEofZK, logEofZFinalK)
	#EofZTotalK = keeton_calculations.getEofZTotalKeeton(EofZK, EofZFinalK) #debug
	logEofZ2TotalK = keeton_calculations.getEofZ2TotalKeetonLog(logEofZ2K, logEofZ2FinalK, logEofZZFinalK)
	#EofZ2TotalK = keeton_calculations.getEofZ2TotalKeeton(EofZ2K, EofZ2FinalK) #debug
	EofLogZTotalK = calculations.calcEofLogZ(logEofZTotalK, logEofZ2TotalK, 'log')
	varLogZTotalK = calculations.calcVarLogZ(logEofZTotalK, logEofZ2TotalK, 'log')
	HKL = keeton_calculations.calcHTotalKeetonLog(logEofZTotalK, np.array(deadPointsLLhood), nLive, nest, livePointsLLhood)
	#HK = keeton_calculations.calcHTotalKeeton(EofZTotalK, np.exp(np.array(deadPointsLLhood)), nLive, nest, np.exp(livePointsLLhood)) #debug
	numSamples = len(totalPointsPhys[:,0])
	if setupDict['verbose']:
		#output.printZHValues(EofZTotal, EofZ2Total, varZ, H, 'linear', 'total', 'recursive') #debug
		#output.printZHValues(EofZFinalK, EofZ2FinalK, varZFinalK, 'not calculated', 'linear', 'final contribution', 'Keeton equations') #debug
		#output.printZHValues(EofZTotalK, EofZ2TotalK, varZTotalK, HK, 'linear', 'total', 'Keeton equations') #debug
		output.printZHValues(logEofZTotal, logEofZ2Total, logVarZTotal, EofLogZTotal, varLogZTotal, H, 'log', 'total', 'recursive')
		output.printZHValues(logEofZFinalK, logEofZ2FinalK, logVarZFinalK, EofLogZFinalK, varLogZFinalK, 'not calculated', 'log', 'final contribution', 'Keeton equations')
		output.printZHValues(logEofZTotalK, logEofZ2TotalK, logVarZTotalK, EofLogZTotalK, varLogZTotalK, HKL, 'log', 'total', 'Keeton equations')
	if setupDict['outputFile']:
		output.writeOutput(setupDict['outputFile'], totalPointsPhys, totalPointsLLhood, logEofWeights, logEofXArr, paramNames ,'log', targetSupport, logEofZTotal, logVarZTotal, EofLogZTotal, varLogZTotal)

def NestedRunLinear(priorFunc, invPriorFunc, LhoodFunc, paramNames, targetSupport, setupDict):
	"""
	function which completes a NS run. parameters of priors and likelihood need to be specified, as well as a flag indication type of prior for each dimension and the pdf for the lhood.
	setupDict contains other setup parameters such as termination type & factor, method of finding new livepoint, details of how weights are calculated, how final Z contribution is added, and directory/file prefix for saved files.
	"""
	nLive = 500 #low value is 50, high value is 500
	nDims = len(paramNames)
	array_checks.checkTargSupShape(targetSupport, nDims)
	livePoints = np.random.rand(nLive, nDims) #initialise livepoints to random values uniformly on [0,1]^D
	livePointsPhys = invPriorFunc(livePoints) #Convert livepoint values to physical values
	array_checks.checkinvPriorShape(livePointsPhys, livePoints.shape)
	livePointsLhood = LhoodFunc(livePointsPhys) #calculate LLhood values of initial livepoints
	livePointsLhood = array_checks.checkLhoodShape(livePointsLhood, nLive) #could also solve this by reshaping array in .pdf method of custom class
	# initialise lists for storing values
	EofXArr = []
	EofWeights = []
	deadPoints = []
	deadPointsPhys = []
	deadPointsLhood = []	
	# initialise mean and variance of Z variables and other moments
	EofZ = 0.
	EofZ2 = 0.
	EofZX = 0.
	EofX = 1.
	EofX2 = 1.
	X = 1.
	# initialise other variables
	LhoodStar = 0.
	H = 0.
	nest = 0
	ZLive = np.inf
	checkTermination = 100
	nonGeomList, boundaryList, geomList, shapeList = geom_sampler.splitGeomParams(setupDict['paramGeomList']) 
	circleList, torusList, sphereList = geom_sampler.splitGeomShapes(geomList, shapeList)
	nonGeomLowerLimits, nonGeomUpperLimits = geom_sampler.getNonGeomLimits(targetSupport, nonGeomList)
	circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits = geom_sampler.getShapeLimits(targetSupport, circleList, torusList, sphereList)
	#begin nested sample loop
	while True:
		LhoodStarOld = LhoodStar 
		deadIndex = np.argmin(livePointsLhood) #index of lowest likelihood livepoint (next deadpoint)
		LhoodStar = livePointsLhood[deadIndex] #LLhood of dead point and new target
		#update expected values of mPriorFuncoments of X and Z, and get posterior weights
		EofZNew, EofZ2, EofZX, EofX, EofX2, EofWeight = recurrence_calculations.updateZnXMoments(nLive, EofZ, EofZ2, EofZX, EofX, EofX2, LhoodStarOld, LhoodStar, setupDict['trapezoidalFlag'])
		EofXArr.append(EofX)
		EofWeights.append(EofWeight)
		H = recurrence_calculations.updateH(H, EofWeight, EofZNew, LhoodStar, EofZ)
		EofZ = EofZNew #update evidence part II
		#WARNING, VIEWING A NUMPY SLICE (IE NOT USING NP.COPY) DOES NOT CREATE A COPY AND SO A-POSTORI CHANGES TO ARRAY WILL AFFECT PREVIOUSLY SLICED ARRAY
		#USE NP.MAY_SHARE_MEMORY(A, B) TO SEE IF ARRAYS SHARE MEMORY, PYTHON 'IS' KEYWORD DOESN'T WORK  
		deadPointPhys = np.copy(livePointsPhys[deadIndex]).reshape(1,-1)
		deadPointsPhys.append(deadPointPhys)
		deadPointLhood = LhoodStar
		deadPointsLhood.append(deadPointLhood)
		#update array where last deadpoint was with new livepoint picked subject to L_new > L*
		if setupDict['sampler'] == 'blind':
			livePointsPhys[deadIndex], livePointsLhood[deadIndex] = samplers.getNewLiveBlind(invPriorFunc, LhoodFunc, LhoodStar, nDims)
		#elif setupDict['sampler'] == 'MH' or setupDict['sampler'] == 'MH geom':
		elif 'MH' in setupDict['sampler']:
			livePointsPhys[deadIndex], livePointsLhood[deadIndex] = samplers.getNewLiveMH(livePointsPhys, deadIndex, priorFunc, targetSupport, LhoodFunc, LhoodStar, nDims, nonGeomList, boundaryList, circleList, torusList, sphereList, nonGeomLowerLimits, nonGeomUpperLimits, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits)
		if setupDict['verbose']:
			output.printUpdate(nest, deadPointPhys, deadPointLhood, EofZ, livePointsPhys[deadIndex].reshape(1, -1), livePointsLhood[deadIndex], 'linear')
		nest += 1
		if nest % checkTermination == 0:
			breakFlag, liveMaxIndex, liveLhoodMax, avLhood, nFinal = ns_loop_funcs.tryTermination(setupDict['verbose'], setupDict['terminationType'], setupDict['terminationFactor'], nest, nLive, EofX, livePointsLhood, LhoodStar, setupDict['ZLiveType'], setupDict['trapezoidalFlag'], EofZ, H)
			if breakFlag: #termination condition was reached
				break
	varZ = calculations.calcVariance(EofZ, EofZ2)
	EofLogZ = calculations.calcEofLogZ(EofZ, EofZ2, 'linear')
	varLogZ = calculations.calcVarLogZ(EofZ, EofZ2, 'linear')
	EofZK, EofZ2K = keeton_calculations.calcZMomentsKeeton(np.array(deadPointsLhood), nLive, nest)
	varZK = calculations.calcVariance(EofZK, EofZ2K)
	EofLogZK = calculations.calcEofLogZ(EofZK, EofZ2K, 'linear')
	varLogZK = calculations.calcVarLogZ(EofZK, EofZ2K, 'linear')
	HK = keeton_calculations.calcHKeeton(EofZK, np.array(deadPointsLhood), nLive, nest)
	if setupDict['verbose']:
		output.printBreak()
		output.printZHValues(EofZ, EofZ2, varZ, EofLogZ, varLogZ, H, 'linear', 'before final', 'recursive')
		output.printZHValues(EofZK, EofZ2K, varZK, EofLogZK, varLogZK, HK, 'linear', 'before final', 'Keeton equations')
	EofZTotal, EofZ2Total, H, livePointsPhysFinal, livePointsLhoodFinal, EofXFinalArr = ns_end_funcs.getFinalContribution(setupDict['verbose'], setupDict['ZLiveType'], setupDict['trapezoidalFlag'], nFinal, EofZ, EofZ2, EofX, EofWeights, H, livePointsPhys, livePointsLhood, avLhood, liveLhoodMax, liveMaxIndex, LhoodStar)
	totalPointsPhys, totalPointsLhood, EofXArr, EofWeights = ns_end_funcs.getTotal(deadPointsPhys, livePointsPhysFinal, deadPointsLhood, livePointsLhoodFinal, EofXArr, EofXFinalArr, EofWeights)
	varZTotal = calculations.calcVariance(EofZTotal, EofZ2Total)
	EofLogZTotal = calculations.calcEofLogZ(EofZTotal, EofZ2Total, 'linear')
	varLogZTotal = calculations.calcVarLogZ(EofZTotal, EofZ2Total, 'linear')
	EofZFinalK, EofZ2FinalK = keeton_calculations.calcZMomentsFinalKeeton(livePointsLhood, nLive, nest)
	varZFinalK = calculations.calcVariance(EofZFinalK, EofZ2FinalK)
	EofLogZFinalK = calculations.calcEofLogZ(EofZFinalK, EofZ2FinalK, 'linear')
	varLogZFinalK = calculations.calcVarLogZ(EofZFinalK, EofZ2FinalK, 'linear')
	EofZZFinalK = keeton_calculations.calcEofZZFinalKeeton(np.array(deadPointsLhood), livePointsLhood, nLive, nest)
	varZTotalK = keeton_calculations.getVarTotalKeeton(varZK, varZFinalK, EofZK, EofZFinalK, EofZZFinalK)
	EofZTotalK = keeton_calculations.getEofZTotalKeeton(EofZK, EofZFinalK)
	EofZ2TotalK = keeton_calculations.getEofZ2TotalKeeton(EofZ2K, EofZ2FinalK, EofZZFinalK)
	EofLogZTotalK = calculations.calcEofLogZ(EofZTotalK, EofZ2TotalK, 'linear')
	varLogZTotalK = calculations.calcVarLogZ(EofZTotalK, EofZ2TotalK, 'linear')
	HK = keeton_calculations.calcHTotalKeeton(EofZTotalK, np.array(deadPointsLhood), nLive, nest, livePointsLhood)
	numSamples = len(totalPointsPhys[:,0])
	if setupDict['verbose']:
		output.printZHValues(EofZTotal, EofZ2Total, varZ, EofLogZTotal, varLogZTotal, H, 'linear', 'total', 'recursive')
		output.printZHValues(EofZFinalK, EofZ2FinalK, varZFinalK, EofLogZFinalK, varLogZFinalK, 'not calculated', 'linear', 'final contribution', 'Keeton equations')
		#print "EofZZFinal (keeton) = %s" %EofZZFinalK
		output.printZHValues(EofZTotalK, EofZ2TotalK, varZTotalK, EofLogZTotalK, varLogZTotalK, HK, 'linear', 'total', 'Keeton equations')
	if setupDict['outputFile']:
		output.writeOutput(setupDict['outputFile'], totalPointsPhys, totalPointsLhood, EofWeights, EofXArr, paramNames ,'linear', targetSupport, EofZTotal, varZTotal, EofLogZTotal, varLogZTotal)
