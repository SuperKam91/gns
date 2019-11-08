#import standard modules 
import numpy as np

#import custom modules
from . import geom

def splitGeomParams(paramGeomList):
	"""
	Get indexes of parameters from paramList
	based on whether parameter should be sampled in 
	physical (non-geometric) or geometric space,
	and add these to two lists which can be used
	to index parameter array 
	"""
	nonGeomList = []
	geomList = []
	shapeList = []
	boundaryList = []
	for i, geo in enumerate(paramGeomList):
		if 'vanilla' in geo or 'wrapped' in geo or 'reflect' in geo:
			nonGeomList.append(i)
			boundaryList.append(geo)
		else:
			geomList.append(i)
			shapeList.append(geo)
	return nonGeomList, boundaryList, geomList, shapeList

def splitGeomShapes(geomList, shapeList):
	"""
	Splits geomList into three separate lists by shape.
	Assumes that if shape in shapeList is torus or sphere, and an element of
	shapeList is one of these shapes, then the next parameter in
	geomList corresponds to second parameter of that pair.
	Hence for this to work, parameters in geomList corresponding to 3-d shapes
	should be paired in order
	"""
	circleList = []
	torusList = []
	sphereList = []
	i = 0
	while i < len(geomList):
		if 'circle' in shapeList[i]:
			circleList.append(geomList[i])
		elif 'torus' in shapeList:
			torusList.append(geomList[i])
			torusList.append(geomList[i+1])
			i += 1
		elif 'sphere' in shapeList:
			sphereList.append(geomList[i])
			sphereList.append(geomList[i+1])
			i += 1
		i += 1
	return circleList, torusList, sphereList

def getNonGeomParams(params, nonGeomList):
	"""
	Return non-geom dimensions of params based on
	vanillaList
	"""
	return params[nonGeomList]

def getGeomParams(params, geomList):
	"""
	Return geometric dimensions of params based on
	geomList. Same as above function
	"""
	return params[geomList]

def getShapeParams(params, circleList, torusList, sphereList):
	"""
	Return different geometric shape dimensions of params based on
	circleList, torusList and sphereList. Same as above function
	"""
	return params[circleList], params[torusList], params[sphereList]

def getNonGeomSigma(sigma, nonGeomList):
	"""
	gets diagonal elements of sigma for non-geom
	parameters based on nonGeomList.
	Array it returns is diagonal
	As above function
	"""
	sigmaArr = np.diag(sigma) 
	#nonGeomSigma = np.diag(sigmaArr[nonGeomList])
	nonGeomSigma = sigmaArr[nonGeomList]
	return nonGeomSigma

def getGeomSigma(sigma, geomList):
	"""
	gets diagonal elements of sigma for geometric
	parameters based on geomList.
	Array it returns is diagonal
	As above function
	"""
	sigmaArr = np.diag(sigma) 
	geomSigma = np.diag(sigmaArr[geomList])
	#geomSigma = sigmaArr[geomList]
	return geomSigma

def getShapeSigma(sigma, circleList, torusList, sphereList):
	"""
	gets diagonal elements of sigma for geometric
	shape parameters based on the three lists.
	Assumes sigma is diagonal or it will miss off-diagonal elements
	Array it returns is not diagonal, as only need diagonal after converting
	to Cartesian coordinates
	"""
	#assumes sigma is diagonal so that this returns a 1-d array of diagonal elements
	sigmaArr = np.diag(sigma) 
	#circleSigma = np.diag(sigmaArr[circeList])
	#torusSigma = np.diag(sigmaArr[torusList])
	#sphereSigma = np.diag(sigmaArr[sphereList])
	circleSigma = sigmaArr[circleList]
	torusSigma = sigmaArr[torusList]
	sphereSigma = sigmaArr[sphereList]
	return circleSigma, torusSigma, sphereSigma

def getNonGeomLimits(targetSupport, nonGeomList):
	"""
	get upper and lower limits of non-geom dimensions from targetSupport upper and lower
	limits.
	"""
	nonGeomLowerLimits = targetSupport[0, nonGeomList]
	nonGeomUpperLimits = targetSupport[1, nonGeomList]
	return nonGeomLowerLimits, nonGeomUpperLimits

def getShapeLimits(targetSupport, circleList, torusList, sphereList):
	"""
	get upper and lower limits and save in separate arrays based on
	geometric shape to be sampled from, from targetSupport upper and lower
	limits.
	"""
	circleLowerLimits = targetSupport[0, circleList] 
	circleUpperLimits = targetSupport[1, circleList]
	torusLowerLimits = targetSupport[0, torusList]
	torusUpperLimits = targetSupport[1, torusList]
	sphereLowerLimits = targetSupport[0, sphereList]
	sphereUpperLimits = targetSupport[1, sphereList]
	return circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits

#def getCartesianCoords(params, sigma, circleList, torusList, sphereList, targetSupport):
def getCartesianCoords(circleArr, torusArr, sphereArr, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits):
	"""
	Takes in arrays of geometrically sampled parameters 
	and returns corresponding points in relevant spaces
	in cartesian coordinates.
	"""
	numCirc = len(circleArr)
	#each torus/ sphere corresponds to two parameters
	numTorus = len(torusArr) // 2
	numSphere = len(sphereArr) // 2
	#circle requires 2-d cartesian coords, torus and sphere requires 3-d
	circCartArr = np.zeros(numCirc * 2)
	torusCartArr = np.zeros(numTorus * 3)
	sphereCartArr = np.zeros(numSphere * 3)
	i = 0
	j = 0
	#for each circular parameter, convert to Cartesian coords and store all of them in 1-d array
	while j < numCirc:
		circCartArr[i], circCartArr[i+1] = geom.point2CartCirc(circleArr[j], circleLowerLimits[j], circleUpperLimits[j]) 
		i += 2
		j += 1
	i = 0
	j = 0
	#convert each pair of torus parameters to Cartesian coords (first of each pair corresponds to phi, second theta)
	while j < 2 * numTorus:
		torusCartArr[i], torusCartArr[i+1], torusCartArr[i+2] = geom.point2CartTorus(torusArr[j], torusArr[j+1], torusLowerLimits[j], torusLowerLimits[j+1], torusUpperLimits[j], torusUpperLimits[j+1]) 
		i += 3
		j += 2
	i = 0
	j = 0
	#convert each pair of sphere parameters to Cartesian coords (first of each pair corresponds to phi, second theta)
	while j < 2 * numSphere:
		sphereCartArr[i], sphereCartArr[i+1], sphereCartArr[i+2] = geom.point2CartSphere(sphereArr[j], sphereArr[j+1], sphereLowerLimits[j], sphereLowerLimits[j+1], sphereUpperLimits[j], sphereUpperLimits[j+1]) 
		i += 3
		j += 2
	return numCirc, numTorus, numSphere, circCartArr, torusCartArr, sphereCartArr

def getCartesianSigma(circleArr, torusArr, sphereArr, circleSigma, torusSigma, sphereSigma, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits):
	"""
	Takes in sigma arrays (1-d, not 2-d diagonal) of geometrically sampled parameters 
	(assumes they're diagonal, off-diagonal elements are missed)
	and returns corresponding sigmas in relevant spaces
	in cartesian coordinates.
	"""
	numCirc = len(circleArr)
	#each torus/ sphere corresponds to two parameters
	numTorus = len(torusArr) // 2
	numSphere = len(sphereArr) // 2
	circCartSigArr = np.zeros(numCirc * 2)
	torusCartSigArr = np.zeros(numTorus * 3)
	sphereCartSigArr = np.zeros(numSphere * 3)
	i = 0
	j = 0
	while j < numCirc:
		circCartSigArr[i], circCartSigArr[i+1] = getCircleSigma(circleArr[j], circleSigma[j], circleLowerLimits[j], circleUpperLimits[j])
		i += 2
		j += 1
	i = 0
	j = 0
	while j < 2 * numTorus:
		torusCartSigArr[i], torusCartSigArr[i+1], torusCartSigArr[i+2] = getTorusSigma(torusArr[j], torusArr[j+1], torusSigma[j], torusSigma[j+1], torusLowerLimits[j], torusLowerLimits[j+1], torusUpperLimits[j], torusUpperLimits[j+1])
		i += 3
		j += 2
	i = 0
	j = 0
	while j < 2 * numSphere:
		sphereCartSigArr[i], sphereCartSigArr[i+1], sphereCartSigArr[i+2] = getSphereSigma(sphereArr[j], sphereArr[j+1], sphereSigma[j], sphereSigma[j+1], sphereLowerLimits[j], sphereLowerLimits[j+1], sphereUpperLimits[j], sphereUpperLimits[j+1])
		i += 3
		j += 2
	return circCartSigArr, torusCartSigArr, sphereCartSigArr

#def getGeomTrialPoint(targetSupport, circleList, torusList, sphereList, numCirc, numTorus, numSphere, circCartArr, torusCartArr, sphereCartArr, circCartSigArr, torusCartSigArr, sphereCartSigArr):
def getGeomTrialPoint(numCirc, numTorus, numSphere, circCartArr, torusCartArr, sphereCartArr, circCartSigArr, torusCartSigArr, sphereCartSigArr, circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits):
	"""
	Takes arrays of Cartesian coords of geometrically sampled parameters
	and gets trial point for each parameter in case of circle, or for each
	pair in case of torus and sphere.
	Returns arrays of physical points (one array for each shape)
	"""
	#circleLowerLimits, circleUpperLimits, torusLowerLimits, torusUpperLimits, sphereLowerLimits, sphereUpperLimits = getShapeLimits(targetSupport, circleList, torusList, sphereList)
	circlePrimeArr = np.zeros(numCirc)
	torusPrimeArr = np.zeros(numTorus * 2)
	spherePrimeArr = np.zeros(numSphere * 2)
	i = 0
	j = 0
	while j < numCirc:
		circlePrimeArr[j] = getCircleTrial(circCartArr[i:i+2], np.diag(circCartSigArr[i:i+2]**2.), circleLowerLimits[j], circleUpperLimits[j])
		i += 2
		j += 1
	i = 0
	j = 0
	while j < 2 * numTorus:
		torusPrimeArr[j], torusPrimeArr[j+1] = getTorusTrial(torusCartArr[i:i+3], np.diag(torusCartSigArr[i:i+3]**2.), torusLowerLimits[j], torusLowerLimits[j+1], torusUpperLimits[j], torusUpperLimits[j+1])
		i += 3
		j += 2
	i = 0
	j = 0
	while j < 2 * numSphere:	
		spherePrimeArr[j], spherePrimeArr[j+1] = getSphereTrial(sphereCartArr[i:i+3], np.diag(sphereCartSigArr[i:i+3]**2.), sphereLowerLimits[j], sphereLowerLimits[j+1], sphereUpperLimits[j], sphereUpperLimits[j+1])
		i += 3
		j += 2
	return circlePrimeArr, torusPrimeArr, spherePrimeArr

def recombineTrialPoint(nonGeomPrimeArr, circlePrimeArr, torusPrimeArr, spherePrimeArr, nonGeomList, circleList, torusList, sphereList):
	"""
	recombine non-geom and geometric dimensions of trial point
	in same order as input parameter
	"""
	trialPoint = np.zeros(len(nonGeomPrimeArr) + len(circlePrimeArr) + len(torusPrimeArr) + len(spherePrimeArr))
	trialList = nonGeomList + circleList + torusList + sphereList
	primeArr = np.concatenate((nonGeomPrimeArr, circlePrimeArr, torusPrimeArr, spherePrimeArr))
	for i, j in enumerate(trialList):
		trialPoint[j] = primeArr[i]
	return trialPoint

def getTwoPiSigma(sigmaP, l, u):
	"""
	Calculates error on phi isin [0, 2pi]
	based on error of p isin [l, u], assuming
	bounds have no errors
	"""
	return sigmaP * 2. * np.pi / (u - l)

def getPiSigma(sigmaP, l, u):
	"""
	Calculates error on phi isin [0, pi]
	based on error of p isin [l, u], assuming
	bounds have no errors
	"""
	return sigmaP * np.pi / (u - l)

def getCircleSigma(p, sigmaP, l, u, method = 'propagation'):
	"""
	If method == 'constant' sets sigmaX = sigmaY = 0.5 (n.b. proposal space is disc radius 2,
	sampling space is unit circle)
	If method == 'propagation' first calls getTwoPiSigma to get error
	on phi and then uses this to get error on
	Cartesian components.
	Has to calculate phi from p which is a little inefficient
	as this is already done when transforming point to Cartesian,
	but should add a massive overhead.
	TODO: rearrange functions so p is converted to phi only once
	"""
	r = 1.
	if method == 'constant':
		sigmaX, sigmaY = 0.5, 0.5
	elif method == 'propagation':
		sigmaPhi = getTwoPiSigma(sigmaP, l, u)
		phi = geom.physPeriod2TwoPi(p, l, u)
		sigmaX, sigmaY = getCircleCartSigma(phi, sigmaPhi, r)	
	return sigmaX, sigmaY

def getCircleCartSigma(phi, sigmaPhi, r):
	"""
	Calculate error on x and y based on 
	error propagation formulae coupled with
	coordinate transformation equations
	"""
	sigmaX = r * np.abs(np.sin(phi)) * sigmaPhi
	sigmaY = r * np.abs(np.cos(phi)) * sigmaPhi
	return sigmaX, sigmaY

def getTorusSigma(p1, p2, sigmaP1, sigmaP2, l1, l2, u1, u2, method = 'propagation'):
	"""
	If method == 'constant' sets sigmaX = sigmaY = sigmaZ = 0.5
	(n.b. proposal space is solid torus greater radius = lesser radius 2, 
	sampling space is surface of torus with greater radius = 2 and lesser radius = 1)
	If method == 'propagation' first calls getTwoPiSigma to get errors on 
	phi & theta and then uses this to get error on
	Cartesian components.
	Has to calculate phi and theta from p1, p2 which is a little inefficient
	as this is already done when transforming point to Cartesian,
	but should add a massive overhead.
	TODO: rearrange functions so p1, p2 are converted to phi and theta only once
	"""
	r = 1.
	R = 2.
	if method == 'constant':
		sigmaX, sigmaY, sigmaZ = 0.5, 0.5, 0.5
	elif method == 'propagation':
		sigmaPhi = getTwoPiSigma(sigmaP1, l1, u1)
		sigmaTheta = getTwoPiSigma(sigmaP2, l2, u2)
		phi = geom.physPeriod2TwoPi(p1, l1, u1)
		theta = geom.physPeriod2TwoPi(p2, l2, u2)
		sigmaX, sigmaY, sigmaZ = getTorusCartSigma(phi, theta, sigmaPhi, sigmaTheta, r, R)	
	return sigmaX, sigmaY, sigmaZ

def getTorusCartSigma(phi, theta, sigmaPhi, sigmaTheta, r, R, sigmaPhiTheta = 0.):
	"""
	Calculate error on x and y based on 
	error propagation formulae coupled with
	coordinate transformation equations
	"""
	sigmaX = np.sqrt((r * np.sin(theta) * np.cos(phi) * sigmaTheta)**2. + ((r * np.cos(theta) * np.sin(phi) + R * np.sin(phi)) * sigmaPhi)**2. + 2. * r * np.sin(theta) * np.cos(phi) * (R * np.sin(phi) + r * np.cos(theta) * np.sin(phi)) * sigmaPhiTheta)
	sigmaY = np.sqrt((r * np.sin(theta) * np.sin(phi) * sigmaTheta)**2. + ((r * np.cos(theta) * np.cos(phi) + R * np.cos(phi)) * sigmaPhi)**2. + 2. * r * np.sin(theta) * np.sin(phi) * (R * np.cos(phi) + r * np.cos(theta) * np.cos(phi)) * sigmaPhiTheta)
	sigmaZ = r * np.abs(np.cos(theta)) * sigmaTheta
	return sigmaX, sigmaY, sigmaZ

def getSphereSigma(p1, p2, sigmaP1, sigmaP2, l1, l2, u1, u2, method = 'constant'):
	"""
	If method == 'constant' sets sigmaX = sigmaY = sigmaZ = 0.01
	(n.b. proposal space is sphere radius = 2, sampling space is surface of sphere radius 1)
	If method == 'propagation' first calls getTwoPiSigma and getPiSigma to get errors on 
	phi & theta and then uses this to get error on
	Cartesian components.
	Has to calculate phi and theta from p1, p2 which is a little inefficient
	as this is already done when transforming point to Cartesian,
	but should add a massive overhead.
	TODO: rearrange functions so p1, p2 are converted to phi and theta only once
	"""
	r = 1.
	if method == 'constant':
		sigmaX, sigmaY, sigmaZ = 0.01, 0.01, 0.01
	elif method == 'propagation':
		sigmaPhi = getTwoPiSigma(sigmaP1, l1, u1)
		sigmaTheta = getPiSigma(sigmaP2, l2, u2)
		phi = geom.physPeriod2TwoPi(p1, l1, u1)
		theta = geom.physPeriod2Pi(p2, l2, u2)
		sigmaX, sigmaY, sigmaZ = getSphereCartSigma(phi, theta, sigmaPhi, sigmaTheta, r)	
	return sigmaX, sigmaY, sigmaZ

def getSphereCartSigma(phi, theta, sigmaPhi, sigmaTheta, r, sigmaPhiTheta = 0.):
	"""
	Calculate error on x and y based on 
	error propagation formulae coupled with
	coordinate transformation equations
	"""
	sigmaX = r * np.sqrt((np.sin(phi) * np.sin(theta) * sigmaPhi)**2. + (np.cos(phi) * np.cos(theta) * sigmaTheta)**2. + 2. * r * np.sin(phi) * np.sin(theta) * r * np.cos(phi) * np.cos(theta) * sigmaPhiTheta)
	sigmaY = r * np.sqrt((np.cos(phi) * np.sin(theta) * sigmaPhi)**2. + (np.sin(phi) * np.cos(theta) * sigmaTheta)**2. + 2. * r * np.cos(phi) * np.sin(theta) * r * np.sin(phi) * np.cos(theta) * sigmaPhiTheta)
	sigmaZ = r * np.abs(np.sin(theta)) * sigmaTheta
	return sigmaX, sigmaY, sigmaZ

def getCircleTrial(mean, cov, l, u):
	"""
	Takes mean and covariance of Cartesian coordinates
	and samples 2-d Gaussian parameterised by these.
	If rBound doesn't evaluate to zero, checks if trial point
	is within radius rBound of origin in x-y plane (in circle radius rBound) 
	"""
	rBound = 2.
	while True:
		xTrial, yTrial = np.random.multivariate_normal(mean, cov)
		if rBound:
			if not checkCircleBounds(xTrial, yTrial, rBound):
				continue
		break
	phiTrial = geom.projectCart2Circ(xTrial, yTrial)
	pTrial = geom.twoPiPeriod2Phys(phiTrial, l, u)
	return pTrial

def checkCircleBounds(x, y, rBound):
	"""
	Check if point given by cartesian coordinates
	is within rBound of centre of circle
	"""
	rho = np.sqrt(x**2. + y**2.)
	if rho > rBound:
		return False
	else:
		return True

def getTorusTrial(mean, cov, l1, l2, u1, u2):
	"""
	Takes mean and covariance of Cartesian coordinates
	and samples 3-d Gaussian parameterised by these.
	If rBound doesn't evaluate to zero, checks if trial point
	is within torus with inner radius = rBound and greater radius = R.
	If rBound != R (for R isin positive real numbers) 
	"""
	R = 2.
	rBound = 2.
	while True:
		xTrial, yTrial, zTrial = np.random.multivariate_normal(mean, cov)
		if rBound:
			if not checkTorusBounds(xTrial, yTrial, zTrial, R, rBound):
				continue
		break
	phiTrial, thetaTrial = geom.projectCart2Torus(xTrial, yTrial, zTrial, R)
	p1Trial = geom.twoPiPeriod2Phys(phiTrial, l1, u1)
	p2Trial = geom.twoPiPeriod2Phys(thetaTrial, l2, u2)
	return p1Trial, p2Trial

def checkTorusBounds(x, y, z, R, rBound):
	"""
	Check if point given in cartesian coordinates
	is within rBound from centre of tube of torus.
	First calculates azimuthal angle of point to get
	coordinate of centre of tube which will be closest
	to the point in question.
	Then calculates the distance from the centre of the tube
	to this point, and checks this is within rBound
	"""
	phi = np.arctan2(y, x)
	phi = geom.switchPolarSys(phi)
	#get point on circle in x-y plane which runs along centre of tube of torus
	xCirc = R * np.cos(phi)
	yCirc = R * np.sin(phi)
	#distance from point to nearest point on centre of tube
	tubeDist = np.sqrt((x - xCirc)**2. + (y - yCirc)**2. + z**2.)
	if tubeDist > rBound:
		return False
	else:
		return True

def getSphereTrial(mean, cov, l1, l2, u1, u2):
	"""
	Takes mean and covariance of Cartesian coordinates
	and samples 3-d Gaussian parameterised by these.
	If rBound doesn't evaluate to zero, checks if trial point
	is within sphere with radius = rBound  
	"""
	rBound = 10.
	while True:
		xTrial, yTrial, zTrial = np.random.multivariate_normal(mean, cov)
		if rBound:
			if not checkSphereBounds(xTrial, yTrial, zTrial, rBound):
				continue
		break
	phiTrial, thetaTrial = geom.projectCart2Sphere(xTrial, yTrial, zTrial)
	p1Trial = geom.twoPiPeriod2Phys(phiTrial, l1, u1)
	p2Trial = geom.piPeriod2Phys(thetaTrial, l2, u2)
	return p1Trial, p2Trial

def checkSphereBounds(x, y, z, rBound):
	"""
	Check if point given by cartesian coordinates
	is within rBound of centre of sphere
	"""
	rho = np.sqrt(x**2. + y**2. + z**2.)
	if rho > rBound:
		return False
	else:
		return True	