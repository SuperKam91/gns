#import standard modules 
import numpy as np
import scipy
try: #newer scipy versions
	from scipy.special import logsumexp
except ImportError: #older scipy versions
	from scipy.misc import logsumexp

#import custom modules
from . import calculations
from . import tools

################Calculate Z moments and H a-posteri using Keeton's methods

def calcZMomentsKeeton(Lhoods, nLive, nest):
	"""
	calculate Z moments a-posteri with full list of Lhoods used in NS loop,
	using equations given in Keeton
	TODO: consider Keeton equations using trapezium rule
	"""
	EofZ = calcEofZKeeton(Lhoods, nLive, nest)
	EofZ2 = calcEofZ2Keeton(Lhoods, nLive, nest)
	return EofZ, EofZ2

def calcEofZKeeton(Lhoods, nLive, nest):
	"""
	Calculate first moment of Z from main NS loop.
	According to paper, this is just E[Z] = 1. / nLive * sum_i^nest L_i * E[t]^i
	"""
	EoftArr = calculations.getEofftArr(calculations.EoftPowi, nLive, nest)
	LEoft = Lhoods * EoftArr
	return 1. / nLive * LEoft.sum()

def calcEofZ2Keeton(Lhoods, nLive, nest):
	"""
	Calculate second (raw) moment of Z from main NS loop (equation 22 Keeton)
	"""
	const = 2. / (nLive * (nLive + 1.))
	summations = calcSums(Lhoods, nLive, nest)
	return const * summations

def calcSums(Lhoods, nLive, nest):
	"""
	Calculate double summation in equation (22) of Keeton using two generator (yielding) functions. 
	First one creates array associated with index of inner sum (which is subsequently summed).
	Second one creates array of summed inner sums, which is then multiplied by array of 
	L_k * E[t]^k terms to give outer summation terms.
	Outer summation terms are added together to give total of double sum.
	"""
	EoftArr = calculations.getEofftArr(calculations.EoftPowi, nLive, nest)
	LEoft = Lhoods * EoftArr
	innerSums = np.fromiter(calcInnerSums(Lhoods, nLive, nest), dtype = float, count = nest)
	outerSums = LEoft * innerSums
	return outerSums.sum()

def calcInnerSums(Lhoods, nLive, nest):
	"""
	Second generator (yielding) function, which returns inner sum for outer index k
	"""
	for k in range(1, nest + 1):
		Eoft2OverEoftArr = calculations.getEofftArr(calculations.Eoft2OverEoftPowi, nLive, k)
		innerTerms = Lhoods[:k] * Eoft2OverEoftArr
		innerSum = innerTerms.sum()
		yield innerSum

def calcSumsLoop(Lhoods, nLive, nest):
	"""
	Calculate double summation in equation (22) of Keeton using double for loop (one for each summation).
	Inefficient (I think) but easy
	"""
	total = 0.
	for k in range(1, nest + 1):
		innerSum = 0.
		for i in range(1, k + 1):
			innerSum += Lhoods[i-1] * calculations.Eoft2OverEoftPowi(nLive, i)
		outerSum = Lhoods[k-1] * calculations.EoftPowi(nLive, k) * innerSum
		total += outerSum
	return total

def calcHKeeton(EofZ, Lhoods, nLive, nest):
	"""
	Calculate H from KL divergence equation transformed to LX space
	as given in Keeton.
	Note this calculates contribution to H from main NS loop
	TODO: consider Keeton equations using trapezium rule
	"""
	sumTerms = Lhoods * np.log(Lhoods) * calculations.getEofftArr(calculations.EoftPowi, nLive, nest)
	sumTerm = 1. / nLive * sumTerms.sum()
	return 1. / EofZ * sumTerm - np.log(EofZ)

def calcZMomentsKeetonLog(LLhoods, nLive, nest):
	"""
	Calculate logs of Z moments based on Keeton's equations
	"""
	logEofZ = calcEofZKeetonLog(LLhoods, nLive, nest)
	logEofZ2 = calcEofZ2KeetonLog(LLhoods, nLive, nest)
	return logEofZ, logEofZ2

def calcEofZKeetonLog(LLhoods, nLive, nest):
	"""
	Calculate log of first moment of Z from main NS loop.
	based on function calcEofZKeeton
	"""
	logEoftArr = np.log(calculations.getEofftArr(calculations.EoftPowi, nLive, nest))
	logLEoft = LLhoods + logEoftArr
	#return tools.logAddArr2(-np.inf, logLEoft) - np.log(nLive)
	return logsumexp(logLEoft) - np.log(nLive)

def calcEofZ2KeetonLog(LLhoods, nLive, nest):
	"""
	calculates logEofZ^2 based on Keeton's equations.
	Based on functions calcSums, calcInnerSums and calcSumsLoop
	"""
	const = np.log(2. / (nLive * (nLive + 1.)))
	logSums = calcLogSums(LLhoods, nLive, nest)
	return const + logSums

def calcLogSums(LLhoods, nLive, nest):
	"""
	Based on calcSums function
	"""
	logEoftArr = np.log(calculations.getEofftArr(calculations.EoftPowi, nLive, nest))
	logLEoft = LLhoods + logEoftArr
	innerLogSums = np.fromiter(calcInnerLogSums(LLhoods, nLive, nest), dtype = float, count = nest)
	outerLogSums = logLEoft + innerLogSums
	#return tools.logAddArr2(-np.inf, outerLogSums)
	return logsumexp(outerLogSums)

def calcInnerLogSums(LLhoods, nLive, nest):
	"""
	based on calcInnerSums function
	"""
	for k in range(1, nest + 1):
		logEoft2OverEoftArr = np.log(calculations.getEofftArr(calculations.Eoft2OverEoftPowi, nLive, k))
		innerLogTerms = LLhoods[:k] + logEoft2OverEoftArr
		#innerLogSum = tools.logAddArr2(-np.inf, innerLogTerms)
		innerLogSum = logsumexp(innerLogTerms)
		yield innerLogSum

def calcLogSumsLoop(LLhoods, nLive, nest):
	"""
	Based on calcSumsLoop function
	"""
	total = -np.inf
	for k in range(1, nest + 1):
		innerLogSum = -np.inf
		for i in range(1, k + 1):
			innerLogSum = np.logaddexp(innerLogSum, LLhoods[i-1] + np.log(calculations.Eoft2OverEoftPowi(nLive, i)))
		outerLogSum = LLhoods[k-1] + np.log(calculations.EoftPowi(nLive, k)) + innerLogSum
		total = np.logaddexp(total, outerLogSum)
	return total

def calcHKeetonLog(logEofZ, LLhoods, nLive, nest):
	"""
	Based on calcHKeeton() function
	Doesn't actually work in log-space, this is to prevent numerical difficulties e.g.
	log(negative number) associated with negative values of LLhoods and logZ (from low L, Z values)
	"""
	LovrZ = np.exp(LLhoods - logEofZ) #hopefully this shouldn't under / over flow
	sumTerms = LovrZ * LLhoods * calculations.getEofftArr(calculations.EoftPowi, nLive, nest)
	sumTerm = 1. / nLive * sumTerms.sum()
	return sumTerm - logEofZ
	
def calcHKeetonLog2(logEofZ, LLhoods, nLive, nest):
	"""
	Based on calcHKeeton() function
	Works in log-space, but will screw up for low values of 
	L and Z since log(L) ~ negative and log(negative) undefined
	"""
	logSumTerms = np.log(LLhoods) + LLhoods + np.log(calculations.getEofftArr(calculations.EoftPowi, nLive, nest))
	#logSumTerm = np.log(1. / logEofZ) + np.log(1. / nLive) + tools.logAddArr2(-np.inf, logSumTerms)
	logSumTerm = np.log(1. / logEofZ) + np.log(1. / nLive) + logsumexp(logSumTerms)
	maxTerm = max(logSumTerm, logEofZ)
	logH = tools.logSubExp(logSumTerm, np.log(logEofZ), logSumTerm)
	return np.exp(logH)
	#return logH

def calcZMomentsFinalKeeton(finalLhoods, nLive, nest):
	"""
	calculate Z moments a-posteri with list of final Lhood points (ones remaining at termination of main loop),
	using equations given in Keeton
	TODO: consider Keeton equations using trapezium rule
	TODO: consider different ways of handling final livepoints (i.e. average over X or max Lhood)
	"""
	EofZ = calcEofZFinalKeeton(finalLhoods, nLive, nest)
	EofZ2 = calcEofZ2FinalKeeton(finalLhoods, nLive, nest)
	return EofZ, EofZ2

def calcEofZFinalKeeton(finalLhoods, nLive, nest):
	"""
	Averages over Lhood, which I don't think is the correct thing to do as it doesn't correspond to a unique parameter vector value.
	However this gives same value for Z as by averaging over X
	TODO: consider other ways of getting final contribution from livepoints with Keeton's method
	"""
	LhoodAv = finalLhoods.mean()
	EofFinalX = calculations.EoftPowi(nLive, nest)
	return EofFinalX * LhoodAv

def calcEofZ2FinalKeeton(finalLhoods, nLive, nest):
	"""
	Averages over Lhood, which I don't think is the correct thing to do as it doesn't correspond to a unique parameter vector value.
	TODO: consider other ways of getting final contribution from livepoints with Keeton's method
	"""
	LhoodAv = finalLhoods.mean()
	EofFinalX2 = calculations.Eoft2Powi(nLive, nest)
	return LhoodAv**2. * EofFinalX2

def calcEofZZFinalKeeton(Lhoods, finalLhoods, nLive, nest):
	"""
	Averages over Lhood for contribution from final points, 
	which I don't think is the correct thing to do as it doesn't correspond to a unique parameter vector value.
	TODO: consider other ways of getting final contribution from livepoints with Keeton's method
	"""
	finalLhoodAv = finalLhoods.mean()
	finalTerm = finalLhoodAv / (nLive + 1.) * calculations.EoftPowi(nLive, nest)
	Eoft2OverEoftArr = calculations.getEofftArr(calculations.Eoft2OverEoftPowi, nLive, nest) 
	loopTerms = Lhoods * Eoft2OverEoftArr
	loopTerm = loopTerms.sum()
	return finalTerm * loopTerm

def calcHTotalKeeton(EofZ, Lhoods, nLive, nest, finalLhoods):
	"""
	Calculates total value of H based on KL divergence equation transformed to 
	LX space as given in Keeton.
	Uses H function used to calculate loop H value (but with total Z), and adapts
	final result to give HTotal
	Note EofZ corresponds to total EofZ
	TODO: consider Keeton equations using trapezium rule
	"""
	LAv = finalLhoods.mean()
	HPartial = calcHKeeton(EofZ, Lhoods, nLive, nest)
	return HPartial + 1. / EofZ * LAv * np.log(LAv) * calculations.EoftPowi(nLive, nest) #n.b. the 2nd contribution isn't HFinal, as we are dividing by EofZTotal not EofZFinal

def calcZMomentsFinalKeetonLog(finalLLhoods, nLive, nest):
	"""
	Calculate log of moments of Z from contribution
	after main NS loop
	"""
	logEofZ = calcEofZFinalKeetonLog(finalLLhoods, nLive, nest)
	logEofZ2 = calcEofZ2FinalKeetonLog(finalLLhoods, nLive, nest)
	return logEofZ, logEofZ2

def calcEofZFinalKeetonLog(finalLLhoods, nLive, nest):
	"""
	Based on function calcEofZFinalKeeton().
	"""
	#logLhoodAv = tools.logAddArr2(-np.inf, finalLLhoods) - np.log(nLive)
	logLhoodAv = logsumexp(finalLLhoods) - np.log(nLive)
	logEofFinalX = np.log(calculations.EoftPowi(nLive, nest))
	return logLhoodAv + logEofFinalX

def calcEofZ2FinalKeetonLog(finalLLhoods, nLive, nest):
	"""
	Based on function calcEofZ2FinalKeetonLog()
	"""
	#logLhoodAv = tools.logAddArr2(-np.inf, finalLLhoods) - np.log(nLive)
	logLhoodAv = logsumexp(finalLLhoods) - np.log(nLive)
	logEofFinalX2 = np.log(calculations.Eoft2Powi(nLive, nest))
	return 2. * logLhoodAv + logEofFinalX2

def calcEofZZFinalKeetonLog(LLhoods, finalLLhoods, nLive, nest):
	"""
	Based on function calcEofZZFinalKeeton()
	"""
	#logFinalLhoodAv = tools.logAddArr2(-np.inf, finalLLhoods) - np.log(nLive)
	logFinalLhoodAv = logsumexp(finalLLhoods) - np.log(nLive)
	finalTerm = logFinalLhoodAv - np.log(nLive + 1.) + np.log(calculations.EoftPowi(nLive, nest))
	logEoft2OverEoftArr = np.log(calculations.getEofftArr(calculations.Eoft2OverEoftPowi, nLive, nest)) 
	loopTerms = LLhoods + logEoft2OverEoftArr
	#loopTerm = tools.logAddArr2(-np.inf, loopTerms)
	loopTerm = logsumexp(loopTerms)
	return finalTerm + loopTerm

def calcHTotalKeetonLog(logEofZ, LLhoods, nLive, nest, finalLLhoods):
	"""
	Based on function calcHTotalKeeton() 
	Again mainly doesn't work in log-space to avoid undefined 
	function evaluations
	"""
	#logFinalLhoodAv = tools.logAddArr2(-np.inf, finalLLhoods) - np.log(nLive)
	logFinalLhoodAv = logsumexp(finalLLhoods) - np.log(nLive)
	HPartial = calcHKeetonLog(logEofZ, LLhoods, nLive, nest)
	LAvOverZ = np.exp(logFinalLhoodAv - logEofZ)
	HPartial2 = LAvOverZ * logFinalLhoodAv * calculations.EoftPowi(nLive, nest)
	return HPartial + HPartial2

def calcHTotalKeetonLog2(logEofZ, LLhoods, nLive, nest, finalLLhoods):
	"""
	Based on function calcHTotalKeeton() 
	Again mainly works in log-space so can mess up for
	small L, Z
	"""
	#logFinalLhoodAv = tools.logAddArr2(-np.inf, finalLLhoods) - np.log(nLive)
	logFinalLhoodAv = logsumexp(finalLLhoods) - np.log(nLive)
	HPartial = calcHKeetonLog2(logEofZ, LLhoods, nLive, nest)
	#logHPartial = calcHKeetonLog2(logEofZ, LLhoods, nLive, nest) 
	logHPartial2 = - logEofZ + logFinalLhoodAv + np.log(logFinalLhoodAv) + np.log(calculations.EoftPowi(nLive, nest))
	H = HPartial + np.exp(logHPartial2)
	return H
	#logH = np.logaddexp(logHPartial, logHPartial2)
	#return logH

#Functions for combining contributions from main NS loop and termination ('final' quantities) for estimate or Z and its error

def getEofZTotalKeeton(EofZ, EofZFinal):
	"""
	get total from NS loop and final contributions
	"""
	return EofZ + EofZFinal

def getEofZ2TotalKeeton(EofZ2, EofZ2Final, EofZZFinal):
	"""
	get total from NS loop and final contributions
	"""
	return EofZ2 + EofZ2Final + 2. * EofZZFinal

def getVarTotalKeeton(varZ, varZFinal, EofZ, EofZFinal, EofZZFinal):
	"""
	Get total variance from NS loop and final contributions.
	For recursive method, since E[ZLive] = E[ZTot] etc., 
	and assuming that the recurrence relations account for the covariance between
	Z and ZFinal, this is just varZFinal.
	For Keeton's method, have to explicitly account for correlation as expectations for Z and ZLive are essentially calculated independently
	TODO: check if recurrence relations of Z and ZFinal properly account for correlation between two  ANSWER: they do not
	"""
	return varZ + varZFinal + 2. * (EofZZFinal - EofZ * EofZFinal)

def getVarTotalKeetonLog(logVarZ, logVarZFinal, logEofZ, logEofZFinal, logEofZZFinal):
	"""
	Get log of total variance based on Keeton's equations
	"""
	#positiveTerms = tools.logAddArr2(-np.inf, np.array([logVarZ, logVarZFinal, np.log(2.) + logEofZZFinal]))
	positiveTerms = logsumexp(np.array([logVarZ, logVarZFinal, np.log(2.) + logEofZZFinal]))
	return tools.logSubExp(positiveTerms, np.log(2.) + logEofZ +  logEofZFinal, positiveTerms)

def getEofZTotalKeetonLog(logEofZ, logEofZFinal):
	"""
	Get log of EofZ total from loop and final contributions
	"""
	return np.logaddexp(logEofZ, logEofZFinal)

def getEofZ2TotalKeetonLog(logEofZ2, logEofZ2Final, logEofZZFinal):
	"""
	Get log of EofZ^2 total from loop and final contributions
	"""
	return logsumexp([logEofZ2, logEofZ2Final, np.log(2.) + logEofZZFinal])
