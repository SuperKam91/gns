#import standard modules 
import numpy as np
import sys

#import custom modules
from . import prob_funcs
from . import geom_sampler

##########Lhood sampling functions

def getNewLiveBlind(invPriorFunc, LhoodFunc, LhoodStar, nDims):
	"""
	Blindly picks points isin U[0, 1]^D, converts these to physical values according to physical prior and uses as candidates for new livepoint until L > L* is found
	LhoodFunc can be Lhood or LLhood func, either works fine
	"""
	trialPointLhood = -np.inf
	while trialPointLhood <= LhoodStar:
		trialPoint = np.random.rand(1, nDims)
		trialPointPhys = invPriorFunc(trialPoint) #Convert trialpoint value to physical value
		trialPointLhood = LhoodFunc(trialPointPhys) #calculate Lhood value of trialpoint
	return trialPointPhys, trialPointLhood	

def getNewLiveMH(livePointsPhys, deadIndex, priorFunc, targetSupport, LhoodFunc, LhoodStar, nDims, nonGeomList, boundaryList, circleList, torusList, sphereList, nonGeomLowerLimits, nonGeomUpperLimits, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits):
	"""
	gets new livepoint using variant of MCMC MH algorithm. From current livepoints (not including one to be excluded in current NS iteration), first picks a point at random as starting point.
	Next the standard deviation of the trial distribution is calculated from the width of the current livepoints (including the one to be excluded) as 0.1 * [max(param_value) - min(param_value)] in each dimension.
	A trial distribution (CURRENTLY GAUSSIAN) is centred on the selected livepoint with the calculated variance, and the output is used as a trial point.
	This point is kept with probability prior(trial) / prior(previous) if L_trial > L* and rejected otherwise.
	Each time a trial point is proposed, nTrials is incremented. If the trial point was accepted nAccept is incremented, if not nReject is.
	The trial distribution is updated based on nAccept, nReject such that the acceptance rate should be roughyl 50%.
	Once nTrials nTrials = maxTrials, if nAccept = 0 the whole process is repeated as failure to do so will mean the returned live point is a copy of another point. If not, the new livepoint is returned.
	One would hope that nAccept is > 1 to ensure that the sampling space is explored uniformly.
	LhoodFunc can be Lhood or LLhood func, either works fine
	TODO: find better function of nDims from which maxTrials can be calculated
	"""
	#proposalType = 'truncated multivariate normal' #used to think this was compulsory if using wrapped parameters. Not so sure anymore
	proposalType = 'multivariate normal'
	nAccept = 0
	maxTrials = 10 * nDims #changed from 20 to 10 on 30/6/18 
	#current deadpoint not a possible starting candidate. This could be ignored
	startCandidates = np.delete(livePointsPhys, (deadIndex), axis = 0)
	trialSigma = calcInitTrialSig(livePointsPhys)
	while nAccept == 0: #ensure that at least one move was made from initially picked point, or new returned livepoint will be same as pre-existing livepoint.
	#in the case of no acceptances, process is started again from step of picking starting livepoint
		#randomly pick starting candidate
		startIndex = np.random.randint(0, len(startCandidates[:,0]))
		startPoint = startCandidates[startIndex]
		nonGeomStartPoint = geom_sampler.getNonGeomParams(startPoint, nonGeomList)
		circleStartPoint, torusStartPoint, sphereStartPoint = geom_sampler.getShapeParams(startPoint, circleList, torusList, sphereList)
		nonGeomSigma = geom_sampler.getNonGeomSigma(trialSigma, nonGeomList)
		circleSigma, torusSigma, sphereSigma = geom_sampler.getShapeSigma(trialSigma, circleList, torusList, sphereList)
		numCirc, numTorus, numSphere, circCartArr, torusCartArr, sphereCartArr = geom_sampler.getCartesianCoords(circleStartPoint, torusStartPoint, sphereStartPoint, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits)
		circCartSigArr, torusCartSigArr, sphereCartSigArr = geom_sampler.getCartesianSigma(circleStartPoint, torusStartPoint, sphereStartPoint, circleSigma, torusSigma, sphereSigma, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits)
		startLhood = LhoodFunc(startPoint) #this is not used per sae, but an arbitrary value is needed for first value in loop. n.b. startLhood is (1,) array not scalar
		nTrials = 0
		nReject = 0
		while nTrials < maxTrials:
			nTrials += 1
			#find physical values of trial point candidate
			#TODO: consider implementing function which gets non-geom components of targetSupport and pass this instead of nonGeomLowerLimits and nonGeomUpperLimits
			nonGeomTrialPoint = pickTrial(nonGeomStartPoint, np.diag(nonGeomSigma**2.), targetSupport[:,nonGeomList], proposalType) #easier to use relevant slice of targetSupport than nonGeomLowerLimits and nonGeomUpperLimits
			#apply boundary conditions on trial point s.t. it has physical values within sampling space domain
			nonGeomTrialPoint = applyBoundaries(nonGeomTrialPoint, targetSupport[:,nonGeomList], boundaryList) #easier to use relevant slice of targetSupport than nonGeomLowerLimits and nonGeomUpperLimits
			circleTrialArr, torusTrialArr, sphereTrialArr = geom_sampler.getGeomTrialPoint(numCirc, numTorus, numSphere, circCartArr, torusCartArr, sphereCartArr, circCartSigArr, torusCartSigArr, sphereCartSigArr, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits)
			trialPoint = geom_sampler.recombineTrialPoint(nonGeomTrialPoint, circleTrialArr, torusTrialArr, sphereTrialArr, nonGeomList, circleList, torusList, sphereList)
			trialLhood = calcTrialLhood(trialPoint, LhoodFunc, targetSupport)
			#returns previous point values if test fails, or trial point values if it passes 
			acceptFlag, startPoint, startLhood = testTrial(trialPoint, startPoint, trialLhood, startLhood, LhoodStar, priorFunc)
			if acceptFlag:
				nAccept += 1
				nonGeomStartPoint = geom_sampler.getNonGeomParams(startPoint, nonGeomList)
				circleStartPoint, torusStartPoint, sphereStartPoint = geom_sampler.getShapeParams(startPoint, circleList, torusList, sphereList)
				numCirc, numTorus, numSphere, circCartArr, torusCartArr, sphereCartArr = geom_sampler.getCartesianCoords(circleStartPoint, torusStartPoint, sphereStartPoint, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits)

			else:
				nReject += 1
			#update trial distribution variance. Doesn't currently do anything
			trialSigma = updateTrialSigma(trialSigma, nAccept, nReject)
			nonGeomSigma = geom_sampler.getNonGeomSigma(trialSigma, nonGeomList)
			circleSigma, torusSigma, sphereSigma = geom_sampler.getShapeSigma(trialSigma, circleList, torusList, sphereList)
			circCartSigArr, torusCartSigArr, sphereCartSigArr = geom_sampler.getCartesianSigma(circleStartPoint, torusStartPoint, sphereStartPoint, circleSigma, torusSigma, sphereSigma, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits)
	return startPoint, startLhood

################MH related functions

def calcInitTrialSig(livePoints):
	"""
	calculate initial standard deviation based on width of domain defined by max and min parameter values of livepoints in each dimension 
	"""
	minParams = livePoints.min(axis = 0)
	maxParams = livePoints.max(axis = 0)
	livePointsWidth = maxParams - minParams
	trialSigma = np.sqrt(np.diag(0.1 * livePointsWidth)) #Sivia 2006 uses 0.1 * domain width
	return trialSigma

def pickTrial(startPoint, trialVar, targetSupport, proposalType):
	"""
	pick trial point based on proposalType, which currently can be 
	multivariate normal (not truncated) or truncated multivariate normal.
	When truncated, the truncation in each dimension is the width of the support
	in that dimension, centred on the startPoint.
	For the dimensions where the support is unbounded, there is no truncation
	"""
	if len(startPoint) == 0: #needed for geometric sampling, when array of vanilla params is empty.
		return startPoint
	if proposalType == 'multivariate normal':
		trialPoint = np.random.multivariate_normal(startPoint, trialVar)
	elif proposalType == 'truncated multivariate normal':
		while True:
			trialPoint = np.random.multivariate_normal(startPoint, trialVar)
			if np.any(trialPoint < (startPoint - 0.5 * targetSupport[2,:])) or np.any(trialPoint > (startPoint + 0.5 * targetSupport[2,:])):
				continue
			else:
				break
	return trialPoint

def testTrial(trialPoint, startPoint, trialLhood, startLhood, LhoodStar, priorFunc):
	"""
	Check if trial point has L > L* and accept with probability prior(trial) / prior(previous)
	"""
	newPoint = startPoint
	newLhood = startLhood
	acceptFlag = False
	if np.isnan(trialLhood): #immediately reject trial point if it gives nan Lhood value
		pass
	elif trialLhood > LhoodStar:
		prob = np.random.rand()
		priorRatio = priorFunc(trialPoint) / priorFunc(startPoint)
		if priorRatio > prob:
			newPoint = trialPoint
			newLhood = trialLhood
			acceptFlag = True
	return acceptFlag, newPoint, newLhood

def updateTrialSigma(trialSigma, nAccept, nReject, method = 'nothing'):
	"""
	update standard deviation as in Sivia 2006, or do nothing.
	Apparently former ensures that ~50% of the points are accepted, but I'm sceptical.
	TODO: research new way to update trial sigma which isn't ridiculous
	"""
	if method == 'nothing': #do nothing
		pass
	elif method == 'sivia':
		if nAccept > nReject:
			trialSigma = trialSigma * np.exp(1. / nAccept)
		else:
		 	trialSigma = trialSigma * np.exp(-1. / nReject)
	return trialSigma

def applyBoundaries(livePoint, targetSupport, boundaryList):
	"""
	Either does nothing, reflects or wraps parameter value at its boundary 
	(give by targetSupport) according to paramGeomList.
	Assumes that values in paramGeomList are sensible, i.e. if the target support
	is unbounded, then its value should be 'nothing'
	"""
	for i in range(len(targetSupport[0,:])):
		#it makes most sense for these two to be the same
		if 'reflect' in boundaryList[i]:
			differenceCorrection = 'reflect'
			pointCorrection = 'reflect'			
		elif 'wrapped' in boundaryList[i]:
			differenceCorrection = 'wrap'
			pointCorrection = 'wrap'
		elif 'vanilla' in boundaryList[i]:
			differenceCorrection = 'nothing'
			pointCorrection = 'nothing'
		else:
			print("invalid value in boundaryList. Exiting...")
			sys.exit(1)
		#if np.isfinite(targetSupport[2,i]):
		#	livePoint[i] = applyBoundary(livePoint[i], targetSupport[0,i], targetSupport[1,i], differenceCorrection, pointCorrection)
		livePoint[i] = applyBoundary(livePoint[i], targetSupport[0,i], targetSupport[1,i], differenceCorrection, pointCorrection)	
	return livePoint

def calcTrialLhood(trialPoint, LhoodFunc, targetSupport):
	"""
	Checks if trialPoint is in support of target function,
	if it is returns trial Lhood value.
	If not returns nan as Lhood value
	"""
	if checkBoundary(trialPoint, targetSupport):
		trialLhood = LhoodFunc(trialPoint)
	else:
		trialLhood = np.nan
	return trialLhood

def checkBoundary(trialPoint, targetSupport):
	"""
	Checks if each dimension of trialPoint is within bounds
	given by targetSupport. If any dimensions are not, returns false.
	Otherwise returns true
	"""
	for i in range(len(targetSupport[0,:])):
		if (trialPoint[i] < targetSupport[0,i]) or (trialPoint[i] > targetSupport[1,i]):
			return False
	return True

def applyBoundary(point, lower, upper, differenceCorrection, pointCorrection):
	"""
	give a point in or outside the domain, in case of point being in the domain it does nothing. 
	When it is outside, it calculates a 'distance' from the domain according to differenceCorrection type, and then uses this 'distance' to transform the point into the domain by using either reflective or wrapping methods according to the value of pointCorrection.
	It is recommended that differenceCorrection and pointCorrection take the same values (makes most intuitive sense to me)
	Examples with domain of -5 to +5:
	wrapping (differenceCorrection == 'wrap' and pointCorrection == 'wrap'):
	-7 -> 3
	-15 -> 5 (n.b. boundaries are effectively treated as open intervals, so -5 wraps round to 5)
	-17 -> 3
	7 -> -3
	15 -> -5
	17 -> -3
	reflecting (differenceCorrection == 'reflect' and pointCorrection == 'reflect'):
	-7 -> -3
	-15 -> -5
	-17 -> 3
	7 -> 3
	15 -> 5
	17 -> -3 
	"""
	#get 'distance' from boundary
	if differenceCorrection == 'wrap':
		pointTemp = modToDomainWrap(point, lower, upper) 
	elif differenceCorrection == 'reflect':
		pointTemp = modToDomainReflect(point, lower, upper)
	elif differenceCorrection == 'nothing':
		pass
	#use 'distance' to reflect or wrap point into domain
	if pointCorrection == 'wrap':
		if point < lower:
			point = upper - pointTemp
		elif point > upper:
			point = lower + pointTemp
	elif pointCorrection == 'reflect':
		if point < lower:
			point = lower + pointTemp
		elif point > upper:
			point = upper - pointTemp
	elif pointCorrection == 'nothing': #set to -inf so Lhood evaluates to zero
		if point < lower or point > upper:
			pass
	return point

#following ensures point isin [lower, upper]. This wraps according to (positive) difference between point and nearest bound and returns a 'distance' which can actually be used to get the correct value of the point within the domain. Effect of this is basically mod'ing the difference by the width of the domain. It makes most intuitive sense to me to use this when you want to wrap the points around the domain.
modToDomainWrap = lambda point, lower, upper : (lower - point) % (upper - lower) if point < lower else (point - upper) % (upper - lower)

def modToDomainReflect(point, lower, upper):
	"""
	following ensures point isin [lower, upper]. This reflects according to (positive) difference between point and nearest bound and returns a 'distance' which can actually be used to get the correct value of the point within the domain. It makes most intuitive sense to me to use this when you want to reflect the points in the domain.
	Operation done to ensure reflecting is different based on whether the difference between the point and the nearest part of the domain is an odd or even (incl. 0) multiple of the boundary.
	"""
	if point < lower:
		outsideMultiple = (lower - point) // (upper - lower) #number of multiples (truncated) of the width of the domain the point lays outside it
		oddFlag = outsideMultiple % 2 #checks if number of multiples of width of domain the point is outside the domain is odd or even (the latter including zero)
		if oddFlag:
			#in this case for a reflective value the mod'd distance needs to be counted from the opposite boundary. This can be done by calculating - delta mod width where delta is difference between closest boundary and point	
			pointTemp = (point - lower) % (upper - lower) 
		else:
			#this is the simpler case in which the reflection is counted from the nearest boundary which is just delta mod width
			pointTemp = (lower - point) % (upper - lower) 
	elif point > upper:
		outsideMultiple = (point - upper) // (upper - lower) #as above but delta is calculated from upper bound
		oddFlag = outsideMultiple % 2 #as above
		if oddFlag:
			#as above
			pointTemp = (upper - point) % (upper - lower) 
		else:
			#as above
			pointTemp = (point - upper) % (upper - lower)
	else: 
		pointTemp = None
	return pointTemp

