#import standard modules 
import numpy as np
import warnings

#import custom modules
from . import tools

#Expected values of f(t) (shrinkage parameter)

def calct(nLive, expectation = 't', sampling = False, maxPoints = False):
	"""
	calc value of t from its pdf, 
	from (supposedely equivalent) way of deriving form of pdf, 
	or from E[.] or E[l(.)]	"""
	if sampling:
		if maxPoints:
			t = np.random.rand(nLive).max()
		else:
			t = np.random.rand() ** (1. / nLive)
	else:
		if expectation == 'logt':
			t = np.exp(-1. / nLive)
		elif expectation == 't':
			t = nLive / (nLive + 1.)
	return t

def calct2(nLive, expectation = 't2', sampling = False, maxPoints = False):
	"""
	calc value of t^2 from its pdf, 
	from (supposedely equivalent) way of deriving form of pdf, 
	or from E[.] or E[l(.)]
	"""
	if sampling:
		if maxPoints:
			##TODO
			pass
		else:
			##TODO
			pass
	else:
		if expectation == 'logt2':
			##TODO
			pass
		elif expectation == 't2':
			t = nLive / (nLive + 2.)
	return t

def calc1mt(nLive, expectation = '1mt', sampling = False, maxPoints = False):
	"""
	calc value of 1-t from its pdf, 
	from (supposedely equivalent) way of deriving form of pdf, 
	or from E[.] or E[l(.)]
	"""
	if sampling:
		if maxPoints:
			##TODO
			pass
		else:
			##TODO
			pass
	else:
		if expectation == 'log1mt':
			##TODO
			pass
		elif expectation == '1mt':
			t = 1. / (nLive + 1.)
	return t

def calc1mt2(nLive, expectation = '1mt2', sampling = False, maxPoints = False):
	"""
	calc value of (1-t)^2 from its pdf, 
	from (supposedely equivalent) way of deriving form of pdf, 
	or from E[.] or E[l(.)]
	"""
	if sampling:
		if maxPoints:
			##TODO
			pass
		else:
			##TODO
			pass
	else:
		if expectation == 'log1mt2':
			##TODO
			pass
		elif expectation == '1mt2':
			t = 2. / ((nLive + 1.) * (nLive + 2.))
	return t

def calcEofts(nLive):
	"""
	calculate expected values of t related variables to update Z and X moments
	"""
	Eoft = calct(nLive)
	Eoft2 = calct2(nLive)
	Eof1mt = calc1mt(nLive)
	Eof1mt2 = calc1mt2(nLive)
	return Eoft, Eoft2, Eof1mt, Eof1mt2

def calcLogEofts(nLive):
	"""
	Calculate log(E[t] - E[t^2]) as it is much easier to do so here than later having on log(E[t]) and log(E[t^2])
	"""
	return np.log(calcEofts(nLive) + (calct(nLive) - calct2(nLive),))

#wrappers around E[t] functions for calculating powers of them.
#required for calculating Z moments with Keeton's method. 

#E[t]^i
EoftPowi = lambda nLive, i : calct(nLive)**i

#E[t^2]^i
Eoft2Powi = lambda nLive, i : calct2(nLive)**i

#(E[t^2]/E[t])^i
Eoft2OverEoftPowi = lambda nLive, i : (calct2(nLive) / calct(nLive))**i

def calcEofftArr(Eofft, nLive, n):
	"""
	Calculates E[f(t)]^i then returns this with yield.
	Yield means next time function is called, 
	it picks off from where it last returned,
	with same variable values as before returning.
	Note the function isn't executed until the generator return by yield is iterated over
	Putting for loop here is faster than filling in blank array
	"""
	for i in range(1, n+1):
		yield Eofft(nLive, i)

def getEofftArr(Eofft, nLive, nest):
	"""
	faster than creating array of zeroes and looping over
	"""
	return np.fromiter(calcEofftArr(Eofft, nLive, nest), dtype = float, count = nest) 

######## calculate/ retrieve final estimates/ errors of Z

def calcVariance(EofX, EofX2):
	"""
	Calculate second moment of X from first moment and raw second moment
	"""
	return EofX2 - EofX**2.

def calcVarianceLog(logEofX, logEofX2):
	"""
	Calc log(var(X)) from log(E[X]) and log(E[X^2])
	Does logsubtractexp manually, so doesn't account for possible underflow issues with exponentiating
	like np.logaddexp() does, but this shouldn't be an issue for the numbers involved here.
	"""
	return tools.logSubExp(logEofX2, 2. * logEofX, logEofX2)

def calcVarZSkillingK(EofZ, nLive, H):
	"""
	Uses definition of error given in Skilling's NS paper, ACCORDING to Keeton.
	Only valid in limit that Skilling's approximation of var[log(Z)] = H / nLive being correct,
	and E[Z]^2 >> var[Z] so that log(1+x)~x approximation is valid.
	Also requires that Z is log-normally distributed
	I think this is only valid for NS loop contributions, not final part or total
	"""
	return EofZ**2. * H / nLive

def calcLogVarZSkillingK(logEofZ, nLive, H):
	"""
	As above but working in log[*] space
	"""
	return 2. * logEofZ + np.log(H) - np.log(nLive)

def calcHSkillingK(EofZ, varZ, nLive):
	"""
	Uses definition of error given in Skilling's NS paper, ACCORDING to Keeton.
	Only valid in limit that Skilling's approximation of var[log(Z)] = H / nLive being correct,
	and E[Z]^2 >> var[Z] so that log(1+x)~x approximation is valid.
	Also requires that Z is log-normally distributed
	I think this is only valid for NS loop contributions, not final part or total
	"""
	return varZ * nLive / EofZ**2.

def calcEofLogZ(EofZ, EofZ2, space = 'linear'):
	"""
	calc E[log(Z)] from E[Z] and E[Z^2] or log(E[Z]) and log(E[Z^2])
	as given in Will's thesis (assumes Z log normal r.v.)
	NOTE in case of Keeton's total value, won't give correct result as won't account
	for covariance between loop and final contributions
	"""
	if space == 'linear':
		return 2. * np.log(EofZ) - 0.5 * np.log(EofZ2)
	elif space == 'log':
		return 2. * EofZ - 0.5 * EofZ2

def calcVarLogZ(EofZ, EofZ2, space = 'linear'):
	"""
	calc var[log(Z)] from E[Z] and E[Z^2] or log(E[Z]) and log(E[Z^2])
	as given in Will's thesis (assumes Z log normal r.v.)
	NOTE in case of Keeton's total value, won't give correct result as won't account
	for covariance between loop and final contributions
	"""
	if space == 'linear':
		return np.log(EofZ2) - 2. * np.log(EofZ)
	elif space == 'log':
		return EofZ2 - 2. * EofZ

def calcVarLogZII(EofZ, varZ, space, method = 'log-normal'):
	"""
	Uses propagation of uncertainty formula or 
	relationship between log-normal r.v.s and the normally distributed log of the log-normal r.v.s
	to calculate var[logZ] from EofZ and varZ (taken from Wikipedia)
	Apart from for Keeton total (where variance doesn't just come from E[Z] and E[Z^2] moments), 
	should be same as first implementation
	"""
	if method == 'uncertainty':
		if space == 'linear':
			return varZ / EofZ**2.
		elif space == 'log':
			return np.log(varZ / EofZ**2.)
	elif method == 'log-normal':
		if space == 'linear':
			return np.log(1. + varZ / (EofZ)**2.)
		elif space == 'log':
			return np.logaddexp(0, varZ - 2. * EofZ)

def calcEofLogZII(EofZ, varZ, space):
	"""
	Calc E[log(Z)] from E[Z] and Var[Z]. Assumes Z is log-normally distributed (taken from Wikipedia)
	Apart from for Keeton total (where variance doesn't just come from E[Z] and E[Z^2] moments), 
	should be same as first implementation
	"""
	if space == 'linear':
		return np.log(EofZ**2. / (np.sqrt(varZ + EofZ**2.)))
	elif space == 'log':
		return EofZ - 0.5 * calcVarLogZII(EofZ, varZ, 'log')

def calcEofZII(EofLogZ, varLogZ, space):
	"""
	calc E[Z] (log(E[Z])) from E[logZ] and var[logZ]. Assumes Z is log-normal (taken from Wikipedia)
	"""
	if space == 'linear':
		return np.exp(EofLogZ + 0.5 * varLogZ)
	elif space == 'log':
		return EofLogZ + 0.5 * varLogZ

def calcVarZII(varLogZ, EofLogZ, EofZ, space, method = 'log-normal'):
	"""
	Uses propagation of uncertainty formula or 
	relationship between log-normal r.v.s and the normally distributed log of the log-normal r.v.s
	to calculate var[Z] (log(var[Z])) from EofZ and varLogZ (taken from Wikipedia)
	"""
	if method == 'uncertainty':
		if space == 'linear':
			return varLogZ * EofZ**2.
		elif space == 'log':
			return np.log(varLogZ * EofZ**2.)
	elif method == 'log-normal':
		if space == 'linear':
			return np.exp(2. * EofLogZ + varLogZ) * (np.exp(varLogZ) - 1.)
		elif space == 'log':
			warnings.filterwarnings("error")
			try:
				ret = 2. * EofLogZ + varLogZ + np.log(np.exp(varLogZ) - 1)
				warnings.filterwarnings("default")
				return ret
			except RunTimeWarning: #assumed to be overflow associated with np.log(np.exp(varLogZ), so assume np.log(np.exp(varLogZ) - 1) ~ varLogZ
				ret =  2. * EofLogZ + varLogZ + varLogZ
				warnings.filterwarnings("default")
				return ret

def calcVarLogZSkilling(H, nLive):
	"""
	Skilling works in log space throughout, including calculating the moments of log(*)
	i.e. E[f(log(*))]. Thus he derives a value for the variance of log(Z), through his discussions of 
	Poisson fluctuations whilst exploring the posterior.
	"""
	return H / nLive

#############DEPRECATED I THINK

def getLogEofXLogEofw(nLive, X):
	"""
	get increment (part of weight for posterior and evidence calculations) based on previous value of X, calculates latest X using t calculated from either expected value or sampling. Expected value can be of t (E[t]) or log(t) E[log(t)]. These are roughly the same for large nLive
	Sampling can take two forms: sampling from the pdf or taking the highest of U[0,1]^Nlive values (from which the pdf form is derived from), so they should in theory be the same.
	"""
	expectation = 't'
	t = calct(nLive, expectation)
	XNew = X * t
	return np.log(XNew), np.log(X - XNew) 

#############DEPRECATED I THINK

def getLogEofWeight(logw, LLhood_im1, LLhood_i, trapezoidalFlag):
	"""
	calculates logw + log(f(L_im1, L_i)) where f(L_im1, L_i) = L_i for standard quadrature
	and f(L_im1, L_i) = (L_im1 + L_i) / 2. for the trapezium rule
	"""
	if trapezoidalFlag:
		return np.log(0.5) + logw + np.logaddexp(LLhood_im1, LLhood_i) #from Will's implementation, Z = sum (X_im1 - X_i) * 0.5 * (L_i + L_im1)
	else:
		return logw + LLhood_i #weight of deadpoint (for posterior) = prior mass decrement * likelihood
