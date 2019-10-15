#import standard modules 
import numpy as np

#import custom modules
from . import calculations

###########Updating expected values of Z, X and H functions (recurrence relations of mine, Handley and Skilling)

def updateZnXMoments(nLive, EofZ, EofZ2, EofZX, EofX, EofX2, LhoodStarOld, LhoodStar, trapezoidalFlag):
	"""
	Wrapper around updateZnXM taking into account whether trapezium rule is used or not
	"""
	if trapezoidalFlag:
		EofZ, EofZ2, EofZX, EofX, EofX2, EofWeight = updateZnXM(nLive, EofZ, EofZ2, EofZX, EofX, EofX2, 0.5 * (LhoodStarOld + LhoodStar))
	else:
		EofZ, EofZ2, EofZX, EofX, EofX2, EofWeight = updateZnXM(nLive, EofZ, EofZ2, EofZX, EofX, EofX2, LhoodStar)
	return EofZ, EofZ2, EofZX, EofX, EofX2, EofWeight

def updateZnXM(nLive, EofZ, EofZ2, EofZX, EofX, EofX2, L):
	"""
	Update moments of Z and X based on their previous values, expected value of random variable t and Lhood value ((L_i + L_i-1) / 2. in case of trapezium rule). 
	Used to calculate the mean and standard deviation of Z, and thus of log(Z) as well 
	TODO: CONSIDER KEETON NON-RECURSIVE METHOD
	"""
	Eoft, Eoft2, Eof1mt, Eof1mt2 = calculations.calcEofts(nLive)
	EofZ, EofWeight = updateEofZ(EofZ, Eof1mt, EofX, L)
	EofZ2 = updateEofZ2(EofZ2, Eof1mt, EofZX, Eof1mt2, EofX2, L)
	EofZX = updateEofZX(Eoft, EofZX, Eoft2, EofX2, L) 
	EofX2 = updateEofX2(Eoft2, EofX2)
	EofX = updateEofX(Eoft, EofX)
	return EofZ, EofZ2, EofZX, EofX, EofX2, EofWeight

def updateEofZ(EofZ, Eof1mt, EofX, L):
	"""
	Update mean estimate of Z.
	"""
	EofWeight = Eof1mt * EofX * L
	return EofZ + EofWeight, EofWeight

def updateEofZX(Eoft, EofZX, Eoft2, EofX2, L):
	"""
	Updates raw 'mixed' moment of Z and X. Required to calculate E(Z)^2.
	"""
	crossTerm = Eoft * EofZX
	X2Term = (Eoft - Eoft2) * EofX2 * L
	return crossTerm + X2Term

def updateEofZ2(EofZ2, Eof1mt, EofZX, Eof1mt2, EofX2, L):
	"""
	Update value of raw 2nd moment of Z based on Lhood value obtained in that NS iteration
	"""
	crossTerm = 2 * Eof1mt * EofZX * L
	X2Term = Eof1mt2 * EofX2 * L**2.
	return EofZ2 + X2Term + crossTerm

def updateEofX2(Eoft2, EofX2):
	"""
	Update value of raw 2nd momement of X
	"""
	return Eoft2 * EofX2

def updateEofX(Eoft, EofX):
	"""
	Update value of raw first moment of X
	"""
	return Eoft * EofX

def updateLogZnXMoments(nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, LLhoodStarOld, LLhoodStar, trapezoidalFlag):
	"""
	as above but for log space
	"""
	if trapezoidalFlag:
		logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight = updateLogZnXM(nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, np.log(0.5) + np.logaddexp(LLhoodStarOld, LLhoodStar))
	else:
		logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight = updateLogZnXM(nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, LLhoodStar)
	return logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight

def updateLogZnXM(nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, LL):
	"""
	as above but for log space
	"""
	logEoft, logEoft2, logEof1mt, logEof1mt2, logEoftmEoft2 = calculations.calcLogEofts(nLive)
	logEofZ, logEofWeight = updateLogEofZ(logEofZ, logEof1mt, logEofX, LL)
	logEofZ2 = updateLogEofZ2(logEofZ2, logEof1mt, logEofZX, logEof1mt2, logEofX2, LL)
	logEofZX = updateLogEofZX(logEoft, logEofZX, logEoftmEoft2, logEofX2, LL) 
	logEofX2 = updateLogEofX2(logEoft2, logEofX2)
	logEofX = updateLogEofX(logEoft, logEofX)
	return logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight

def updateLogEofZ(logEofZ, logEof1mt, logEofX, LL):
	"""
	as above but for log space
	"""
	logEofWeight = logEof1mt + logEofX + LL 
	return np.logaddexp(logEofWeight, logEofZ), logEofWeight

def updateLogEofZX(logEoft, logEofZX, logEoftmEoft2, logEofX2, LL):
	"""
	as above but for log space
	"""
	crossTerm = logEoft + logEofZX
	X2Term = logEoftmEoft2 + logEofX2 + LL
	return np.logaddexp(crossTerm, X2Term)

def updateLogEofZ2(logEofZ2, logEof1mt, logEofZX, logEof1mt2, logEofX2, LL):
	"""
	as above but for log space
	"""
	crossTerm = np.log(2) + logEof1mt + logEofZX + LL
	X2Term = logEof1mt2 + logEofX2 + 2. * LL
	newTerm = np.logaddexp(crossTerm, X2Term)
	return np.logaddexp(logEofZ2, newTerm)

def updateLogEofX2(logEoft2, logEofX2):
	"""
	as above but for log space
	"""
	return logEoft2 + logEofX2

def updateLogEofX(logEoft, logEofX):
	"""
	as above but for log space
	"""
	return logEoft + logEofX

def updateZnXMomentsFinal(nFinal, EofZ, EofZ2, EofX, Lhood_im1, Lhood_i, trapezoidalFlag, errorEval):
	"""
	Wrapper around updateZnXMomentsF taking into account whether trapezium rule is used or not
	"""
	if trapezoidalFlag:
		EofZ, EofZ2, EofWeight = updateZnXMomentsF(nFinal, EofZ, EofZ2, EofX, (Lhood_im1 + Lhood_i) / 2., errorEval)		
	else:
		EofZ, EofZ2, EofWeight = updateZnXMomentsF(nFinal, EofZ, EofZ2, EofX, Lhood_i, errorEval)
	return EofZ, EofZ2, EofWeight

def updateZnXMomentsF(nFinal, EofZ, EofZ2, EofX, L, errorEval):
	"""
	TODO: rewrite docstring
	TODO: CONSIDER KEETON NON-RECURSIVE METHOD WHICH EXPLICITLY ACCOUNTS FOR CORRELATION BETWEEN EOFZ AND EOFZLIVE
	"""
	if errorEval == 'recursive':
		EofX = updateEofXFinal(EofX, nFinal)
		EofX2 = updateEofX2Final(EofX, nFinal)
		EofZ2 = updateEofZ2Final(EofZ2, EofX, EofZ, EofX2, L)
		EofZ, EofWeight = updateEofZFinal(EofZ, EofX, L)
	return EofZ, EofZ2, EofWeight

def updateEofZ2Final(EofZ2, EofX, EofZ, EofX2, L):
	"""
	TODO: rewrite docstring
	"""
	crossTerm = 2 * EofX * EofZ * L
	XTerm = EofX2 * L**2.
	return EofZ2 + crossTerm + XTerm

def updateEofZFinal(EofZ, EofX, L):
	"""
	TODO: rewrite docstring
	"""
	EofWeight = EofX * L
	return EofZ + EofWeight, EofWeight

def updateEofXFinal(EofX, nFinal):
	"""
	can't be proved mathematically, X is treated deterministically to be X / nLive
	"""
	return EofX / nFinal

def updateEofX2Final(EofX, nFinal):
	"""
	can't be proved mathematically, just derived from recurrence relations
	"""
	return EofX**2. / nFinal**2.

def updateLogZnXMomentsFinal(nFinal, logEofZ, logEofZ2, logEofX, LLhood_im1, LLhood_i, trapezoidalFlag, errorEval):
	"""
	Wrapper around updateZnXMomentsF taking into account whether trapezium rule is used or not
	"""
	if trapezoidalFlag:
		logEofZ, logEofZ2, logEofWeight = updateLogZnXMomentsF(nFinal, logEofZ, logEofZ2, logEofX, np.log(0.5) + np.logaddexp(LLhood_im1, LLhood_i), errorEval)		
	else:
		logEofZ, logEofZ2, logEofWeight = updateLogZnXMomentsF(nFinal, logEofZ, logEofZ2, logEofX, LLhood_i, errorEval)
	return logEofZ, logEofZ2, logEofWeight

def updateLogZnXMomentsF(nFinal, logEofZ, logEofZ2, logEofX, LL, errorEval):
	"""
	TODO: rewrite docstring
	"""
	if errorEval == 'recursive':
		logEofX = updateLogEofXFinal(logEofX, nFinal)
		logEofX2 = updateLogEofX2Final(logEofX, nFinal)
		logEofZ2 = updateLogEofZ2Final(logEofZ2, logEofX, logEofZ, logEofX2, LL)
		logEofZ, logEofWeight = updateLogEofZFinal(logEofZ, logEofX, LL)
	return logEofZ, logEofZ2, logEofWeight

def updateLogEofZ2Final(logEofZ2, logEofX, logEofZ, logEofX2, LL):
	"""
	TODO: rewrite docstring
	"""
	crossTerm = np.log(2.) + logEofX + logEofZ + LL
	XTerm = logEofX2 + 2. * LL
	newTerm = np.logaddexp(crossTerm, XTerm)
	return np.logaddexp(logEofZ2, newTerm)

def updateLogEofZFinal(logEofZ, logEofX, LL):
	"""
	TODO: rewrite docstring
	"""
	logEofWeight = logEofX + LL
	return np.logaddexp(logEofZ, logEofWeight), logEofWeight

def updateLogEofXFinal(logEofX, nFinal):
	"""
	can't be proved mathematically, just derived from recurrence relations
	"""
	return logEofX - np.log(nFinal)

def updateLogEofX2Final(logEofX, nFinal):
	"""
	can't be proved mathematically, just derived from recurrence relations
	"""
	return 2. * logEofX - 2. * np.log(nFinal)

def updateH(H, weight, ZNew, Lhood, Z):
	"""
	Same as Skilling's implementation but in linear space
	Handles FloatingPointErrors associated with taking np.log(0) (0 * log(0) = 0)
	TODO: consider trapezium rule for sum and derive equivalent for recurrence relation
	I'm not sure this is correct as Skilling's implementation uses E[log(Z)], E[log(t)] so H is actually
	exp(E[log(weight)] - E[log(ZNew)]) * log(Lhood) + exp(E[log(Z)] - E[log(ZNew)]) * (H + E[log(Z)]) - E[log(ZNew)].
	However, it is in very close (to within millionths of a percent) with the version derived from Keeton's paper
	"""
	np.seterr(all = 'raise')
	try:
		H = 1. / ZNew * weight * np.log(Lhood) + Z / ZNew * (H + np.log(Z)) - np.log(ZNew)
	except FloatingPointError: #take lim Z->0^+ Z / ZNew * (H + log(Z)) = 0
		H =  1. / ZNew * weight * np.log(Lhood) - np.log(ZNew)
	np.seterr(all = 'warn')
	return H

def updateHLog(H, logWeight, logZNew, LLhood, logZ): 
	"""
	update H using previous value, previous and new log(Z) and latest weight
	Isn't a non-log version as H propto log(L).
	As given in Skilling's paper.
	100 percent accurate implementation should be as explained above.
	TODO: consider if trapezium rule should lead to different implementation
	"""
	np.seterr(all = 'raise')
	try:
		H = np.exp(logWeight - logZNew) * LLhood + np.exp(logZ - logZNew) * (H + logZ) - logZNew
	except FloatingPointError: #when logZ is -infinity, np.exp(logZ) * logZ cannot be evaluated. Treat it as zero, ie treat it as lim Z->0^+ exp(logZ) * logZ = 0
		H = np.exp(logWeight - logZNew) * LLhood - logZNew
	np.seterr(all = 'warn')
	return H
