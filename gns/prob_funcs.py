#import standard modules 
import numpy as np
import sys
import scipy.stats 
from scipy.special import ive as ModifiedBessel, gamma as Gamma #for Kent distribution
try: #newer scipy versions
	from scipy.special import logsumexp
except ImportError: #older scipy versions
	from scipy.misc import logsumexp
import inspect

#import custom modules

"""
NB scipy.stats.norm uses standard deviation, scipy.stats.multivariate_normal uses covariance!!
NB if x.shape = (m,1), then scipy.stats.rv_continuous.pdf(x) returns array shape (m, 1) 
whereas scipy.stats.multivariate_normal.pdf(x) returns array shape (m,)
"""

###########PDF related functions

def fitPriors(priorParams):
	"""
	Only currently handles one dimensional (independent priors). Scipy.stats multivariate functions do not have built in inverse CDF methods, so if I want to consider multivariate priors I may have to write my own code.
	Note scipy.stats.uniform takes parameters loc and scale where the boundaries are defined to be
	loc and loc + scale, so scale = upper - lower bound.
	Returns list of fitted prior objects length of nDims (one function for each parameter).

	The prior types (integers) correspond as follows:

	1: Uniform

	2: Gaussian

	3: Sine
	"""
	priorFuncs = []
	priorFuncsPpf = []
	priorFuncsLogPdf = []
	priorType = priorParams[0,:]
	param1Vec = priorParams[1,:]
	param2Vec = priorParams[2,:]
	for i in range(len(priorType)):
		if priorType[i] == 1:
			priorFunc = scipy.stats.uniform(param1Vec[i], param2Vec[i] - param1Vec[i])
		elif priorType[i] == 2:
			priorFunc = scipy.stats.norm(param1Vec[i], param2Vec[i])
		elif priorType[i] == 3:
			priorFunc = scipy.stats.sine()
		else:
			print("priors other than uniform, Gaussian and sin not currently supported")
			sys.exit(1)
		priorFuncs.append(priorFunc)
	return priorFuncs

def getPriorPdfs(priorObjs):
	"""
	Takes list of fitted prior objects, returns list of objects' .pdf() methods
	"""
	priorFuncsPdf = []
	for obj in priorObjs:
		priorFuncsPdf.append(obj.pdf)
	return priorFuncsPdf

def getPriorLogPdfs(priorObjs):
	"""
	Takes list of fitted prior objects, returns list of objects' .logpdf() methods
	"""
	priorFuncsLogPdf = []
	for obj in priorObjs:
		priorFuncsLogPdf.append(obj.logpdf)
	return priorFuncsLogPdf

def getPriorPpfs(priorObjs):
	"""
	Takes list of fitted prior objects, returns list of objects' .ppf() methods
	"""
	priorFuncsPpf = []
	for obj in priorObjs:
		priorFuncsPpf.append(obj.ppf)
	return priorFuncsPpf

def invPrior(livePoints, priorFuncsPpf):
	"""
	take in array of livepoints each which has value isin[0,1], and has nDim dimensions. Output physical array of values corresponding to priors for each parameter dimension.
	"""
	livePointsPhys = np.zeros_like(livePoints)
	for i in range(len(priorFuncsPpf)):
		livePointsPhys[:,i] = priorFuncsPpf[i](livePoints[:,i])
	return livePointsPhys

def priorFuncsProd(livePoint, priorFuncsPdf):
	"""
	calculates pdf of prior for each parameter dimension, then multiplies these together to get the pdf of the prior (i.e. the prior pdf assuming the parameters are independent)
	Works in linear space, but can easily be adapted if this ever leads to underflow errors (consider sum of log(pi(theta))).
	Only currently designed to work with one livepoint at a time
	"""
	livePointPriorValues = np.zeros_like(livePoint)
	for i in range(len(priorFuncsPdf)):
		livePointPriorValues[i] = priorFuncsPdf[i](livePoint[i])
	priorProd = livePointPriorValues.prod()
	return priorProd

def logPriorFuncsSum(livePoint, priorFuncsLogPdf):
	"""
	calculates logpdf of prior for each parameter dimension,
	then adds these together to get the logpdf of the prior (i.e. the prior logpdf assuming the parameters are independent)
	"""
	livePointPriorValues = np.zeros_like(livePoint)
	for i in range(len(priorLogFuncsPdf)):
		livePointPriorLogValues[i] = priorFuncsLogPdf[i](livePoint[i])
	logPriorSum = livePointPriorLogValues.sum()
	return logPriorSum

class priorObjs:
	"""
	class with priorFuncsP*f as attributes required when creating an object, with methods
	which evaluate these functions (and multiply together in case of pdf) whilst just requiring
	livepoints as argument.
	This is essentially a work around for the toys models so the prior and inverse prior functions can be called
	with just livepoints as an argument, to fit to the way a user specified prior would be best called
	"""

	def __init__(self, priorFuncsPdf, priorFuncsLogPdf, priorFuncsPpf):
		"""
		get pdf and ppf methods of scipy.stats.* objects
		"""
		self.priorFuncsPdf = priorFuncsPdf 
		self.priorFuncsLogPdf = priorFuncsLogPdf
		self.priorFuncsPpf = priorFuncsPpf

	def invPrior(self, livePoints):
		"""
		Same as the function invPrior(), but uses
		self.priorFuncsPpf rather than taking it as an argument
		"""
		livePointsPhys = np.zeros_like(livePoints)
		for i in range(len(self.priorFuncsPpf)):
			livePointsPhys[:,i] = self.priorFuncsPpf[i](livePoints[:,i])
		return livePointsPhys

	def priorFuncsProd(self, livePoint):
		"""
		Same as the function priorFuncsProd(), but uses
		self.priorFuncsPdf rather than taking it as an argument
		Only currently designed to work with one livepoint at a time.
		Here we have opposite problem to LhoodObjs class. When using theoretical Z/ H functions,
		parameter has shape (1, nDims) when it needs to have shape (nDims) to not cause an IndexError exception
		"""
		livePointPriorValues = np.zeros_like(livePoint)
		for i in range(len(self.priorFuncsPdf)):
			try:
				livePointPriorValues[i] = self.priorFuncsPdf[i](livePoint[0,i])
			except IndexError:
				livePointPriorValues[i] = self.priorFuncsPdf[i](livePoint[i])
		priorProd = livePointPriorValues.prod()
		return priorProd

	def logPriorFuncsSum(self, livePoint):
		"""
		Same as the function logPriorFuncsSum(), but uses
		self.priorFuncsPdf rather than taking it as an argument.
		Here we have opposite problem to LhoodObjs class. When using theoretical Z/ H functions,
		parameter has shape (1, nDims) when it needs to have shape (nDims) to not cause an IndexError exception
		"""
		livePointPriorLogValues = np.zeros(len(self.priorFuncsLogPdf))
		for i in range(len(self.priorFuncsLogPdf)):
			try:
				livePointPriorLogValues[i] = self.priorFuncsLogPdf[i](livePoint[0,i])
			except IndexError:
				livePointPriorLogValues[i] = self.priorFuncsLogPdf[i](livePoint[i])
		logPriorSum = livePointPriorLogValues.sum()
		return logPriorSum

class LhoodObjs:
	"""
	Class which is essentially defined to wrap around scipy.stats.continuous_rv inheriting classes
	(e.g. scipy.stats.norm).
	Main purpose of class is to have methods .pdf (.logpdf) which evaluate the .pdf methods of a list of 
	rv_continuous instances (specified by LhoodTypes) and multiply (add) together the resulting values.
	Useful so that you can create an object whose .pdf method evaluates product of Lhoods and can still be used by
	Lhood() (LLhood())
	Note the outputted Lhood has shape (len(x[:]),), NOT AS IN CASE OF instances of single continuous_rv instances. 
	This is because the variable Lhoods is declared to have this shape.
	"""

	def __init__(self, mu, sigma, LhoodTypes, bounds = np.array([])):
		"""
		set values of attributes and call method which fits
		Lhoods
		"""
		self.mu = mu
		self.sigma = sigma
		self.LhoodTypes = LhoodTypes
		self.LhoodObjsList = []
		self.bounds = bounds
		self.fitLhoods()		

	def fitLhoods(self):
		"""
		Separately fits 1-d likelihood functions (normal or von Mises)
		using each element of self.mu, and each diagonal element of self.sigma.
		NB von Mises is parameterised by kappa = 1 / variance, normal is parameterised by standard deviation
		"""
		for i, LhoodType in enumerate(self.LhoodTypes):
			if LhoodType == 'normal':
				LhoodObj = scipy.stats.norm(loc = self.mu[i], scale = self.sigma[i,i])
			elif LhoodType == 'von mises':
				LhoodObj = scipy.stats.vonmises(loc = self.mu[i], kappa = 1. / self.sigma[i,i]) #got rid of sigma^2 on 21/01/18. n.b. this is variance not sigma for this case
			elif LhoodType == 'truncated normal':
				#scale of truncated normal affects bounds of truncation such that you must set bounds = bounds_true * 1 / scale
				a = self.bounds[i,0] * 1. / self.sigma[i,i]
				b = self.bounds[i,1] * 1. / self.sigma[i,i]
				LhoodObj = scipy.stats.truncnorm(a = a, b = b, loc = self.mu[i], scale = self.sigma[i,i])
			elif LhoodType == 'uniform':
				LhoodObj = scipy.stats.uniform(loc = self.mu[i], scale =  self.sigma[i,i] - self.mu[i])
			elif LhoodType == 'kent':
				self.mu = self.mu.reshape(3,3) #reshape gamma matrix to 3x3 here
				g1 = self.mu[0,:]
				g2 = self.mu[1,:]
				g3 = self.mu[2,:]
				kappa = self.sigma[0]
				beta = self.sigma[1]
				LhoodObj = KentDistribution(kappa, beta, g1, g2, g3)
			elif LhoodType == 'kent sum':
				g1s = self.mu[3*i]
				g2s = self.mu[3*i+1]
				g3s = self.mu[3*i+2]
				kappas = self.sigma[2*i]
				betas = self.sigma[2*i+1]
				LhoodObj = KentDistributionSum(kappas, betas, g1s, g2s, g3s)
			elif LhoodType == 'gauss kent sum': #bit messy but can't get consistent with for loop without using another index
				LhoodObjsList2 = []
				gaussLhoodObj = scipy.stats.multivariate_normal(self.mu[0], self.sigma[0])
				LhoodObjsList2.append(gaussLhoodObj)
				g1s = self.mu[1]
				g2s = self.mu[2]
				g3s = self.mu[3]
				kappas = self.sigma[1]
				betas = self.sigma[2]
				kentLhoodObj = KentDistributionSum(kappas, betas, g1s, g2s, g3s)
				LhoodObjsList2.append(kentLhoodObj)
				LhoodObj = LhoodObjsList2 #so line outside for loop doesn't need to be changed
			self.LhoodObjsList.append(LhoodObj)

	def pdf(self, x):
		"""
		evaluates .pdf method of all Lhood objects in self.LhoodObjsList
		and multiplies resulting values together.
		It may be the case that x is a 1d array (n,) instead of a 2d array (m,n), this is what the
		IndexError except statements are for. Could instead ensure that all Lhood vectors are
		instead arrays, but I don't think it makes much difference in terms of efficiency or clarity of code.
		Note this isn't in an issue in integrating functions, as they reshape each parameter point to (1,nDims)
		"""
		try:
			Lhoods = np.array([1.]*len(x[:,0]))
		except IndexError: 
			Lhoods = np.array([1.])
		for i, LhoodObj in enumerate(self.LhoodObjsList):
			if 'angles' in inspect.getargspec(LhoodObj.logpdf)[0]: #kent distribution requires two-dim array as argument
				try:
					Lhoods *= LhoodObj.pdf(x[:,2*i:2*i+2]) #assumes each pair (in 2nd dimension) corresponds to two arguments required for kent
				except IndexError:
					Lhoods *= LhoodObj.pdf(x[2*i:2*i+2])
			else:	
				try:
					Lhoods *= LhoodObj.pdf(x[:,i])
				except IndexError:
					Lhoods *= LhoodObj.pdf(x[i])
		return Lhoods

	def logpdf(self, x):
		"""
		evaluates .logpdf method of all Lhood objects in self.LhoodObjsList
		and adds resulting values together.
		It may be the case that x is a 1d array (n,) instead of a 2d array (m,n), this is what causes the IndexErrors. Could instead ensure that all Lhood vectors are
		instead arrays, but I don't think it makes much difference in terms of efficiency or clarity of code.
		Note this isn't in an issue in integrating functions, as they reshape each parameter point to (1,nDims)
		This could be done by writing separate .pdf methods for when x is 1-d or 2-d
		"""
		#this is messy but sufficient for what we're doing. n.b. this is only for evaluating lhood, so doesn't affect actual implementation of gns itself
		try:
			LLhoods = np.zeros_like(x[:,0], dtype = np.float) #need to specify dtype or it uses int32
		except IndexError: 
			LLhoods = np.array([0.])
		for i, LhoodObj in enumerate(self.LhoodObjsList):
			try:
				if 'angles' in inspect.getargspec(LhoodObj.logpdf)[0]: #kent distribution requires two-dim array as argument
					try:
						LLhoods += LhoodObj.logpdf(x[:,2*i:2*i+2]) #assumes each pair (in 2nd dimension) corresponds to two arguments required for kent
					except IndexError:
						LLhoods += LhoodObj.logpdf(x[2*i:2*i+2])
				else:
					try:
						LLhoods += LhoodObj.logpdf(x[:,i])
					except IndexError:
						LLhoods += LhoodObj.logpdf(x[i])
			except AttributeError: #LhoodObj is in fact a list of LhoodObjs (gauss and sphere model)
				#if 'multivariate_normal' in str(self.LhoodObjsList[0,0]):  #this shouldn't be needed unless I try functions other than gaussian
				#another hack for the multivariate norm, which requires more than one dimension of nDims
				#p.s. this is a MASSIVE hack, requires gauss lhood obj to be before kent
				gaussDim = 20
				try:
					LLhoods += LhoodObj[0].logpdf(x[:,0:gaussDim])
				except IndexError:
					LLhoods += LhoodObj[0].logpdf(x[0:gaussDim])
				try:
					LLhoods += LhoodObj[1].logpdf(x[:,gaussDim:])
				except IndexError:
					LLhoods += LhoodObj[1].logpdf(x[gaussDim:])
		return LLhoods

def fitLhood(LhoodParams):
	"""
	fit scipy.stats objects (without data) for parameters to make future evaluations much faster.
	LLhoodType values correspond as follows

	2: is multivariate Gaussian, applicable in most 'usual' circumstances

	3: is the 1d von Mises distribution, a 'wrapped likelihood function defined on [-pi, pi],
	equivalent to having a likelihood defined on the unit circle and parameterised by theta isin [-pi, pi]
	
	4: is the 2d (independent) von Mises distribution, a wrapped likelihood function defined on [-pi, pi] x [-pi, pi], 
	equivalent to having a likelihood defined on the unit torus and parameterised by theta isin [-pi, pi] and phi isin [-pi, pi]
	
	5: uniform x truncated Gaussian on [0, pi]
	
	6: von Mises on [-pi, pi] x truncated Gaussian on [-pi/2, pi/2]
	
	7: von Mises on [-pi, pi] x truncated Gaussian on [0, pi]
	
	8: von Mises on [-pi, pi]^4
	
	9: von Mises on [-pi, pi]^6
	
	10: Kent distribution on a sphere
	
	11: sum of Kent distributions on a sphere
	
	12: sums of Kent distributions on three spheres
	
	13: sums of Kent distributions on five spheres
	
	14: sums of Kent distributions on six spheres
	
	15: Gaussian x sum of Kent distributions on a sphere
	
	16: von Mises on [-pi, pi]^8
	
	17: von Mises on [-pi, pi]^10
	"""
	LhoodType = LhoodParams[0]
	if LhoodParams[0] < 11 or LhoodParams[0] == 16 or LhoodParams[0] == 17: #hack 
		mu = LhoodParams[1].reshape(-1)
	else:
		mu = LhoodParams[1]
	sigma = LhoodParams[2]
	if LhoodType == 2:
		LhoodObj = scipy.stats.multivariate_normal(mu, sigma)
	elif LhoodType == 3:
		LhoodObj = scipy.stats.vonmises(loc = mu, kappa = np.reciprocal(sigma))
	elif LhoodType == 4:
		LhoodTypes = ['von mises', 'von mises']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)
	elif LhoodType == 5:
		LhoodTypes = ['uniform', 'truncated normal']
		bounds = np.array([0., 0., 0., np.pi]).reshape(2,2)
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes, bounds)
	elif LhoodType == 6:
		LhoodTypes = ['von mises', 'truncated normal']
		bounds = np.array([0., 0., -np.pi / 2., np.pi / 2.]).reshape(2,2)
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes, bounds)	
	elif LhoodType == 7:
		LhoodTypes = ['von mises', 'truncated normal']
		bounds = np.array([0., 0., 0., np.pi]).reshape(2,2)
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes, bounds)		
	elif LhoodType == 8: #four-torus
		LhoodTypes = ['von mises', 'von mises', 'von mises', 'von mises']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)		
	elif LhoodType == 9: #six-torus
		LhoodTypes = ['von mises', 'von mises', 'von mises', 'von mises', 'von mises', 'von mises']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)		
	elif LhoodType == 10: #Kent distribution
		LhoodTypes = ['kent']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)	#mu is 3x3 gamma matrix, sigma is kappa and beta	
	elif LhoodType == 11: #Kent distribution sum
		LhoodTypes = ['kent sum']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)	#mu is a list (3,) of list of gamma vectors, sigma is lists of kappas and betas	
	elif LhoodType == 12: #Kent distribution sum on multiple spheres
		LhoodTypes = ['kent sum', 'kent sum', 'kent sum']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)	
	elif LhoodType == 13: 		
		LhoodTypes = ['kent sum', 'kent sum', 'kent sum', 'kent sum', 'kent sum']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)	
	elif LhoodType == 14: 
		LhoodTypes = ['kent sum', 'kent sum', 'kent sum', 'kent sum', 'kent sum', 'kent sum']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)	
	elif LhoodType == 15: 
		LhoodTypes = ['gauss kent sum']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)	
	elif LhoodType == 16: #eight-torus
		LhoodTypes = ['von mises', 'von mises', 'von mises', 'von mises', 'von mises', 'von mises', 'von mises', 'von mises']
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)	
	elif LhoodType == 17: #eight-torus
		LhoodTypes = ['von mises'] * 10
		LhoodObj = LhoodObjs(mu, sigma, LhoodTypes)		
	return LhoodObj

def Lhood(LhoodObj):
	"""
	Returns .pdf method of LhoodObj 
	"""
	return LhoodObj.pdf

def LLhood(LhoodObj):
	"""
	Returns .logpdf method of LhoodObj 
	"""
	return LhoodObj.logpdf

def cartesian_from_spherical(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def infinite_sum(f, i0 = 0, eps = 1e-8):
    i = i0
    total = 0.
    while True:
        diff = f(i)
        if diff < eps * total:
            return total
        total += diff
        i += 1

class KentDistribution():
	"""
	Originally written by Will Handley, edited by Kamran Javid.
	Attempts to roughly follow style of scipy.stats functions
	in terms of how the pdf is fitted and evaluated
	"""
	def __init__(self, kappa, beta, g1, g2, g3):
	    self.kappa = kappa
	    self.beta = beta
	    self.c = self.cCalc()
	    self.g1 = g1
	    self.g2 = g2
	    self.g3 = g3

	def cCalc(self):
		"""
		Calculate normalisation coefficient c
		"""
		c = infinite_sum(lambda j: Gamma(j + 1. / 2.)/Gamma(j + 1) * self.beta**(2. * j) * (self.kappa / 2.)**(-2. * j - 1./2.) * ModifiedBessel(2. * j + 1. / 2., self.kappa))
		return c

	def logpdf(self, angles):
		"""
		Uses try and IndexError block for same reason as pdf and logpdf methods above
		"""
		if self.kappa < 0:
		    raise ValueError("KentDistribution: Parameter kappa ({}) must be >= 0".format(self.kappa))
		elif self.beta < 0:
		    raise ValueError("KentDistribution: Parameter beta ({}) must be >= 0".format(self.beta))
		elif 2 * self.beta > self.kappa:
		    raise ValueError("KentDistribution: Parameter beta ({}) must be <= kappa / 2 ({})".format(self.beta, self.kappa / 2.))
		try:
			x = cartesian_from_spherical(angles[:,0], angles[:,1])
		except IndexError:
			x = cartesian_from_spherical(angles[0], angles[1])
		gx1 = x.transpose().dot(self.g1).transpose()
		gx2 = x.transpose().dot(self.g2).transpose()
		gx3 = x.transpose().dot(self.g3).transpose()
		return - np.log(self.c) + self.kappa * gx1 + self.beta * (gx2**2. - gx3**2.)

	def pdf(self, angles):
		return np.exp(self.logpdf(angles))

class KentDistributionSum():
	"""
	Sum of Kent distribution pdfs.
	"""
	def __init__(self, kappas, betas, g1s, g2s, g3s):
		self.kentList = []
		for i in range(len(kappas)):
			self.kentList.append(KentDistribution(kappas[i], betas[i], g1s[i], g2s[i], g3s[i]))

	def logpdf(self, angles):
		Lpdfs = np.array([kentObj.logpdf(angles) for kentObj in self.kentList])
		LpdfSums = []
		try:
			LpdfSums = np.array([logsumexp(Lpdfs[:,i]) for i in range(len(Lpdfs[0,:]))])
		except IndexError:
			LpdfSums = logsumexp(Lpdfs)
		return LpdfSums

	def pdf(self, angles):
		return np.exp(self.logpdf(angles))
