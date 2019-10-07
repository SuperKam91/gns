#import standard modules 
import numpy as np

#import custom modules
import recurrence_calculations
import output

############final contribution to NS sampling functions

def getFinalContributionLog(verbose, ZLiveType, trapezoidalFlag, nFinal, logEofZ, logEofZ2, logEofX, logEofWeights, H, livePointsPhys, livePointsLLhood, avLLhood, liveLLhoodMax, liveMaxIndex, LLhoodStar, errorEval = 'recursive'):
	"""
	Get final contribution from livepoints after NS loop has ended. Way of estimating final contribution is dictated by ZLiveType.
	Also updates H value and gets final weights (and physical values) for posterior
	this function could be quite taxing on memory as it has to copy all arrays/ lists across
	NOTE: for standard quadrature summation, average Lhood and average X give same values of Z (averaging over X is equivalent to averaging over L). However, correct posterior weights are given by latter method, and Z errors are different in both cases
	"""
	livePointsLLhood = checkIfAveragedLhood(nFinal, livePointsLLhood, avLLhood) #only relevant for 'average' ZLiveType
	if 'average' in ZLiveType:
		LLhoodsFinal = np.concatenate((np.array([LLhoodStar]), livePointsLLhood))
		logEofZOld = logEofZ
		logEofZ2Old = logEofZ2
		for i in range(nFinal): #add weight of each remaining live point incrementally so H can be calculated easily (according to formulation given in Skilling)
			logEofZLive, logEofZ2Live, logEofWeightLive = recurrence_calculations.updateLogZnXMomentsFinal(nFinal, logEofZOld, logEofZ2Old, logEofX, LLhoodsFinal[i], LLhoodsFinal[i+1], trapezoidalFlag, errorEval)
			logEofWeights.append(logEofWeightLive)
			H = recurrence_calculations.updateHLog(H, logEofWeightLive, logEofZLive, LLhoodsFinal[i+1], logEofZOld)
			logEofZOld = logEofZLive
			logEofZ2Old = logEofZ2Live
			if verbose:
				output.printFinalLivePoints(i, livePointsPhys[i], LLhoodsFinal[i+1], ZLiveType, 'log')
		livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr = getFinalAverage(livePointsPhys, livePointsLLhood, logEofX, nFinal, avLLhood, 'log')
	elif ZLiveType == 'max Lhood': #assigns all remaining prior mass to one point which has highest likelihood (of remaining livepoints)
		logEofZLive, logEofZ2Live, logEofWeightLive = recurrence_calculations.updateLogZnXMomentsFinal(nFinal, logEofZ, logEofZ2, logEofX, LLhoodStar, liveLLhoodMax, trapezoidalFlag, errorEval)
		logEofWeights.append(logEofWeightLive) #add scalar to list (as in 'average' case) instead of 1 element array
		H = recurrence_calculations.updateHLog(H, logEofWeightLive, logEofZLive, liveLLhoodMax, logEofZ)
		livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr = getFinalMax(liveMaxIndex, livePointsPhys, liveLLhoodMax, logEofX)
		if verbose:
			output.printFinalLivePoints(liveMaxIndex, livePointsPhysFinal, livePointsLLhoodFinal, ZLiveType, 'log')
	return logEofZLive, logEofZ2Live, H, livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr

def getFinalContribution(verbose, ZLiveType, trapezoidalFlag, nFinal, EofZ, EofZ2, EofX, EofWeights, H, livePointsPhys, livePointsLhood, avLhood, liveLhoodMax, liveMaxIndex, LhoodStar, errorEval = 'recursive'):
	"""
	as above but in linear space
	"""
	livePointsLhood = checkIfAveragedLhood(nFinal, livePointsLhood, avLhood) 
	if (ZLiveType == 'average Lhood') or (ZLiveType == 'average X'):
		EofZOld = EofZ
		EofZ2Old = EofZ2
		LhoodsFinal = np.concatenate((np.array([LhoodStar]), livePointsLhood))
		for i in range(nFinal): 
			EofZLive, EofZ2Live, EofWeightLive = recurrence_calculations.updateZnXMomentsFinal(nFinal, EofZOld, EofZ2Old, EofX, LhoodsFinal[i], LhoodsFinal[i+1], trapezoidalFlag, 'recursive')
			EofWeights.append(EofWeightLive)
			H = recurrence_calculations.updateH(H, EofWeightLive, EofZLive, LhoodsFinal[i+1], EofZOld)
			EofZOld = EofZLive
			EofZ2Old = EofZ2Live
			if verbose:
				output.printFinalLivePoints(i, livePointsPhys[i], LhoodsFinal[i+1], ZLiveType, 'linear')
		livePointsPhysFinal, livePointsLhoodFinal, EofXFinalArr = getFinalAverage(livePointsPhys, livePointsLhood, EofX, nFinal, avLhood, 'linear')
	elif ZLiveType == 'max Lhood': 
		EofZLive, EofZ2Live, EofWeightLive = recurrence_calculations.updateZnXMomentsFinal(nFinal, EofZ, EofZ2, EofX, LhoodStar, liveLhoodMax, trapezoidalFlag, 'recursive')
		EofWeights.append(EofWeightLive) 
		H = recurrence_calculations.updateH(H, EofWeightLive, EofZLive, liveLhoodMax, EofZ)
		livePointsPhysFinal, livePointsLhoodFinal, EofXFinalArr = getFinalMax(liveMaxIndex, livePointsPhys, liveLhoodMax, EofX)
		if verbose:
			output.printFinalLivePoints(liveMaxIndex, livePointsPhysFinal, livePointsLhoodFinal, ZLiveType, 'linear')
	return EofZLive, EofZ2Live, H, livePointsPhysFinal, livePointsLhoodFinal, EofXFinalArr

def checkIfAveragedLhood(nFinal, livePointsLhood, avLhood):
	"""
	Checks if Lhood was averaged over in getLogWeightsLive or not.
	If it was, need to work with average LLhood value for remainder of calculations,
	if not then carry on working with nLive size array of LLhoods.
	For ZLiveType == 'max', average is just taken over array size one with max Lhood value in it, so it is still just max value
	"""
	if nFinal == 1:
		return avLhood
	else:
		return livePointsLhood

def getFinalAverage(livePointsPhys, livePointsLLhood, X, nFinal, avLLhood, space):
	"""
	gets final livepoint values and X value per remaining livepoint for average Z criteria
	NOTE Xfinal is a list not a numpy array
	space says whether you are working in linear or log space (X or logX)
	"""
	livePointsPhysFinal = getLivePointsPhysFinal(livePointsPhys, avLLhood) #only relevant for 'average' ZLiveType
	livePointsLLhoodFinal = livePointsLLhood
	if space == 'linear':
		Xfinalarr = [X / nFinal] * nFinal
	else:
		Xfinalarr = [X - np.log(nFinal)] * nFinal
	return livePointsPhysFinal, livePointsLLhoodFinal, Xfinalarr

def getLivePointsPhysFinal(livePointsPhys, avLhood):
	"""
	Get physical values associated with remaining contribution of livepoints. If LLhood isn't averaged over (X is) this is just the input livepoint values, but if LLhood is averaged it is non-trivial, I.E. THE PHYSICAL VALUES ASSOCIATED WITH THIS POINT ARE MEANININGLESS
	Only relevant for ZLiveType == 'average' as for 'max' case, physical values are just that corresponding to max(L)
	These are needed for posterior samples of remaining contribution of livepoints
	"""
	if not avLhood: #ZLiveType == 'max' means livePointsPhys is already just one livepoint, averageLhoodOrX == 'average X' means retain previous array
		return livePointsPhys
	else: #need to obtain one livepoint from set of nLive. NO (KNOWN AT STAGE OF ALGORITHM) PHYSICAL VECTOR CORRESPONDS TO THIS LIKEILIHOO, SO THIS VALUE IS MEANINGLESS
		return livePointsPhys.mean(axis = 0).reshape(1,-1)

def getFinalMax(liveMaxIndex, livePointsPhys, liveLhoodMax, X):
	"""
	get livepoint and physical livepoint values
	corresponding to maximum likelihood point in remaining
	livepoints.
	Note Xfinal is a list not a numpy array or a scalar
	Function works for log or linear space
	"""
	livePointsPhysFinal = livePointsPhys[liveMaxIndex].reshape(1,-1)
	livePointsLhoodFinal = np.array([liveLhoodMax]) #for consistency with 'average' equivalent function
	Xfinal = [X] #for consistency with 'average' equivalent function, make it a list.
	return livePointsPhysFinal, livePointsLhoodFinal, Xfinal

###############final datastructure / output functions

def getTotal(deadPointsPhys, livePointsPhysFinal, deadPointsLhood, livePointsLhoodFinal, XArr, XFinalArr, weights):
	"""
	gets final arrays of physical, llhood and X values for all accepted points in algorithm.
	This function mutates deadPointsPhys by appending numpy array livePointsPhysFinal. This is at the end of the program
	however, so it shouldn't be an issue.
	Concatenate works on a list of numpy arrays (those corresponding to deadPoints should have shape (1, nDims) and there should be nest of them,
	the single numpy arrays corresponding to the final live points should have shape (nLive, nDims) if average of Z was used for final contribution or (1, nDims) if max of Z was used.
	Concatenating list of numpy arrays is much more efficient than using np.append() at each iteration. 
	"""
	deadPointsPhys.append(livePointsPhysFinal)
	totalPointsPhys = np.concatenate(deadPointsPhys)
	totalPointsLhood = np.append(deadPointsLhood, livePointsLhoodFinal)
	XArr = np.append(XArr, XFinalArr)
	weights = np.array(weights)
	return totalPointsPhys, totalPointsLhood, XArr, weights