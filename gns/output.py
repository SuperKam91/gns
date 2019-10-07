#import standard modules 
import numpy as np
import sys
import scipy

#import custom modules
import samplers
import tools

########file output functions

#summary file containing most information of sampled points
writeSummary = lambda outputFile, params, Lhood, weights, XArr, header : np.savetxt(outputFile + '_summary.txt', np.column_stack((params, Lhood, weights, XArr)), delimiter = ',', header = header)

#chains file in format needed for getDist: importance weight (weights or logWeights), LHood (Lhood or LLhood), phys param values
writeTxt = lambda outputFile, weights, LLhood, params : np.savetxt(outputFile + '.txt', np.column_stack((weights, LLhood, params)))

writeTheorZ = lambda Z, ZErr, outputFile : np.savetxt(outputFile + '_tz.txt', np.array([Z, ZErr])) 

def writeParamNames(outputFile, paramNames):
	"""
	Write file giving index and list parameter names for getDist
	"""
	nameFile = open(outputFile + '.paramnames', 'w')
	for i, name in enumerate(paramNames):
		nameFile.write('p%i %s\n' %(i+1, name))
	nameFile.close()

def rectifyLigoParamNames(file):
	"""
	paramNames in ligo runs I've already done aren't in Latex. 
	So Latex them in the .paramnames file for getdist here
	"""
	f = open(file, 'r')
	fStr = f.read()
	f.close()
	fStr = fStr.replace('phi', '\\phi')
	fStr = fStr.replace('theta', '\\theta')
	fStr = fStr.replace('Phi_c', '\\phi_c')
	f = open(file, 'w')
	f.write(fStr)
	f.close()

def rectifyShapeParamNames(file):
	"""
	paramNames in shape toy models which I've already ran aren't Latex'd. 
	So Latex them in the .paramnames file for getdist here
	"""
	f = open(file, 'r')
	fStr = f.read()
	f.close()
	fStr = fStr.replace('phi', '\\phi')
	fStr = fStr.replace('theta', '\\theta')
	f = open(file, 'w')
	f.write(fStr)
	f.close()

def writeRanges(outputFile, paramNames, targetSupport):
	"""
	write file with hard constraints on parameter boundaries. 
	N means that constraints are inferred from data
	Here constraints are derived from target function's support (sampling space), and unbounded
	supports are assigned N N
	"""
	rangeFile = open(outputFile + '.ranges', 'w')
	for i in range(len(paramNames)):
		if np.isfinite(targetSupport[2,i]):
			rangeFile.write('p%i %s %s\n' %(i+1, targetSupport[0,i], targetSupport[1,i]))
		else:
			rangeFile.write('p%i N N\n' %(i+1))
	rangeFile.close()

def writeOutput(outputFile, totalPointsPhys, totalPointsLhood, weights, XArr, paramNames, space, targetSupport, Z, varZ, lnZ, lnVarZ):
	"""
	writes a summary file which contains values for all sampled points.
	Also writes files needed for getDist.
	When inputs are log values, the weights written are transformed to linear space, in order for KDE to work.
	Furthermore these weights are normalised by dividing by Z
	For log case, Z and varZ should actually be ln(E[Z]) and ln(var[Z])
	"""
	paramNamesStr = ', '.join(paramNames)
	if space == 'linear':
		summaryStr = ' Lhood, E[weights], E[X], E[Z] = %s, var[Z] = %s, E[ln(Z)] = %s, var[ln(Z)] = %s' %(Z, varZ, lnZ, lnVarZ)
		LLhood = np.log(totalPointsLhood)
		normWeights = weights / Z
	else: #everything is given in log space
		summaryStr = ' LLhood, lnE[Weights], ln(E[X]), ln(E[Z]) = %s, ln(var[Z]) = %s, E[ln(Z)] = %s, var[ln(Z)] = %s' %(Z, varZ, lnZ, lnVarZ)
		LLhood = totalPointsLhood
		normWeights = np.exp(weights - Z)
	header = paramNamesStr + summaryStr
	writeSummary(outputFile, totalPointsPhys, totalPointsLhood, normWeights, XArr, header)
	writeTxt(outputFile, normWeights, LLhood, totalPointsPhys)
	writeParamNames(outputFile, paramNames)
	writeRanges(outputFile, paramNames, targetSupport)

def writeTheoreticalSamples(outputFile, logPriorFunc, invPriorFunc, LLhoodFunc, targetSupport, paramNames, method, priorHyperParams = None):
	"""
	Write file of theoretical values of posterior in format of getdist .txt file to be used in getdist
	method == 'sampling' samples parameter space according to prior, then evaluates LLhood at these points.
	method == 'grid' forms a nDims-dimensional mesh grid and evaluates posterior at each point on this grid.
	Obviously infeasible for high dimensions. In this case, the 'LLhood' value written to getdist file is actually LLhood + logPrior. Again all weights are 1.
	Size of grid determined by domain of sampling space. If prior is unbounded (presumably Gaussian), take width in that dimension to be 5 standard deviations each side of prior mean
	Appends '_theor_s' or '_theor_g' to name of file where it saves results.
	TO MAKE THIS WORK IN GETDIST, WEIGHTS HAVE TO BE PROPORTIONAL TO LHOOD IN CASE OF SAMPLING METHOD, OR POSTERIOR IN CASE OF GRID METHOD. HENCE WE SET THE WEIGHTS EQUAL TO THE LHOOD OR POSTERIOR, AND SET THE LHOOD TO 1 (LLHOOD = 0)
	Will probably only work if n is high
	priorHyperParams is a (2. nDims) array with the hyperparameters (e.g. mean and standard dev) of the prior in each dimension. Only needed if usings the gridding method, and one or more of the priors is unbounded, to determine upper and lower bounds of grid
	"""
	nDims = len(paramNames) 
	if method == 'sampling':
		outputFile = outputFile + '_ts'
		n = int(1e4) #total number of points
		uniformPoints = np.random.rand(n, nDims)
		params = invPriorFunc(uniformPoints) 
		LLhood = LLhoodFunc(params)
	if method == 'grid':
		outputFile = outputFile + '_tg'
		oneDn = 1000 #number of points per dimension
		n = oneDn ** nDims
		oneDGrids = []	
		for i in range(len(targetSupport[0,:])):
			if np.isfinite(targetSupport[2,i]):
				lowerBound = targetSupport[0,i]
				upperBound = targetSupport[1,i]	
			else:
				mu = priorHyperParams[0,i]
				sigma = priorHyperParams[1,i]
				lowerBound = mu - 10 * sigma
				upperBound = mu + 10 * sigma
			oneDGrids.append(np.linspace(lowerBound, upperBound, oneDn))
		meshGrids = np.meshgrid(*oneDGrids)
		params = np.hstack((meshGrid.reshape(-1,1) for meshGrid in meshGrids))
		logPrior = np.zeros(n)
		for i in range(n): #there should be a better way of doing this. But as it stands, logPriorFunc only works on (nDim,) or (1, nDim) arrays
			logPrior[i] = logPriorFunc(params[i,:])
		LLhood = LLhoodFunc(params).reshape(-1,)
		logLPrior = logPrior + LLhood
		LLhood = logLPrior #THIS ISN'T JUST LLHOOD, JUST SET TO THIS NAME SO SAME writeTxt() function call can be made for both cases
	###################################
	#THIS IS NECESSARY FOR SAMPLES TO WORK IN GETDIST BY LOADING IN SAMPLES	
	#LLhoodTotSum = tools.logAddArr2(-np.inf, LLhood)
	LLhoodTotSum = scipy.misc.logsumexp(LLhood)
	weights = np.exp(LLhood - LLhoodTotSum) #hopefully this shouldn't cause underflow, but if it does I don't think it can be avoided
	#if it does cause underflow, could try another normalising factor e.g. max(LLhood) or arbitrary value
	LLhood = np.array([0.] * n).reshape(-1,1)
	###################################
	writeTxt(outputFile, weights, LLhood, params)
	writeParamNames(outputFile, paramNames)
	writeRanges(outputFile, paramNames, targetSupport)

###################print output functions

def printUpdate(nest, deadPointPhys, deadPointLhood, EofZ, livePointPhys, livePointLhood, space):
	"""
	gives update on latest deadpoint and newpoint found to replace it
	"""
	if space == 'log':
		L = 'LLhood'
		Z = 'ln(E[Z])'
	elif space == 'linear':
		L = 'Lhood'
		Z = 'E[Z]'
	else:
		print "invalid space"
		sys.exit(1)
	print "for deadpoint %i: physical value = %s %s value = %s" %(nest, deadPointPhys, L, deadPointLhood)
	print "%s = %s" % (Z, EofZ)
	print "new live point obtained: physical value = %s %s has value = %s" %(livePointPhys, L, livePointLhood)

def printBreak():
	"""
	tell user final contribution to sampling is being calculated
	"""
	print "adding final contribution from remaining live points"

def printZHValues(EofZ, EofZ2, varZ, lnZ, lnVarZ, H, space, stage, method):
	"""
	print values of Z (including varios moments, variance) and H
	in either log or linear space, at a given stage and calculated by a given method
	If using log space, EofZ, EofZ2, varZ should actually be ln(E[Z]), ln(E[Z^2]) and ln(var[Z]) 
	"""
	lnEZ = 'E[ln(Z)]'
	lnVar = 'var[ln(Z)]'
	if space == 'log':
		Z = 'ln(E[Z])'
		Z2 = 'ln(E[Z^2])'
		var = 'ln(var[Z])'
	elif space == 'linear':
		Z = 'E[Z]'
		Z2 = 'E[Z^2]'
		var = 'var[Z]'
	else:
		print "invalid space"
		sys.exit(1)
	print "%s %s (%s) = %s" %(Z, stage, method, EofZ)
	print "%s %s (%s) = %s" %(Z2, stage, method, EofZ2)
	print "%s %s (%s) = %s" %(var, stage, method, varZ)
	print "%s %s (%s) = %s" %(lnEZ, stage, method, lnZ)
	print "%s %s (%s) = %s" %(lnVar, stage, method, lnVarZ)
	print "H %s (%s) = %s" %(stage, method, H)

def printTheoretical(ZTheor, ZTheorErr, HTheor, HTheorErr):
	"""
	Outputs values for theoretical values of Z and H (and their errors)
	"""
	print "Z_Theor = %s" %ZTheor
	print "Z_TheorErr = %s" %ZTheorErr
	print "H_Theor = %s" %HTheor 
	print "H_TheorErr = %s" %HTheorErr

def printSampleNum(numSamples):
	"""
	Print number of samples used in sampling (including final livepoints used for posterior weights)
	"""
	print "total number of samples = %i" %numSamples

def printTerminationUpdateInfo(nest, terminator):
	"""
	Print update on termination status when evaluating by H value
	"""
	print "current end value is %i. Termination value is %f" %(nest, terminator)

def printTerminationUpdateZ(EofZLive, endValue, terminationFactor, space):
	"""
	Print update on termination status when evaluating by Z ratio
	"""
	if space == 'linear':
		Z = 'E[Z_Live]'
	elif space == 'log':
		Z = 'ln(E[Z_Live])'
	else:
		print "invalid space"
		sys.exit(1)
	print "%s = %s" %(Z, EofZLive)
	print "current end value is %s. Termination value is %s" %(endValue, terminationFactor)

def printFinalLivePoints(i, physValue, Lhood, ZLiveType, space):
	"""
	print information about final livepoints used to calculate final
	contribution to Z/ posterior samples.
	"""
	if space == 'linear':
		L = 'Lhood'
	elif space == 'log':
		if ZLiveType == 'average Lhood':
			L = 'ln(average Lhood)'
		else:	
			L = 'LLhood'
	else:
		print "invalid space"
		sys.exit(1)
	if ZLiveType == 'average Lhood':
		print "'average' physical value = %s (n.b. this has no useful meaning), %s = %s" %(physValue, L, Lhood)
	elif ZLiveType == 'average X':
		print "remaining livepoint number %i: physical value = %s %s value = %s" %(i, physValue, L, Lhood)
	elif ZLiveType == 'max Lhood':
		print "maximum %s remaining livepoint number %i: physical value = %s %s value = %s" %(L, i, physValue, L, Lhood)