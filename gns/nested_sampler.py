import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats
import scipy.integrate

# import getdist

# misc functions


def logAddArr(x, y, axis=None):
    """
    logaddexp where x is a scalar and y is an array.
    Returns a scalar in case that axis = None (exponentiates elements of array then adds them together).
    np implementation on its own returns an array i.e. doesn't sum exponentiated elements of array.
    Axis specifies axis to do summation of exponentials over, and if specified returns an array with shape of the remaining dimensions.
    """
    yExp = np.exp(y)
    logySum = np.log(yExp.sum(axis=axis))
    return np.logaddexp(x, logySum)


def logAddArr2(x, y, indexes=(None, )):
    """
    Alternative version of logAddArr that avoids over/ underflow errors of exponentiating the array y, to the same extent that np.logaddexp() does
    Note however that it is slower than logAddArr, so in cases where over/ underflow isn't an issue, use that
    Loops over each specified element of y (using rowcol values) and adds to log of sums.
    By default loops over entire array, but to specify a certain row for e.g. a 2d array set indexes to (row_index, slice(None))
    or for a certain column (slice(None), col_index)
    """
    result = x
    for l in np.nditer(y[indexes]):
        result = np.logaddexp(result, l)
    return result


def logsubexp(a, b):
    """
    Only currently works for scalars.
    Calculates log(exp(a) - exp(b)).
    Subtracts max(a,b) from a and b before exponentiating
    in attempt to avoid underflow when a and b are small
    """
    maxab = np.max(a, b)
    expa = np.exp(a - maxab)
    expb = np.exp(b - maxab)
    return np.log(expa - expb)


# PDF related functions


def fitPriors(priorParams):
    """
    Only currently handles one dimensional (independent priors). Scipy.stats multivariate functions do not have built in inverse CDF methods, so if I want to consider multivariate priors I may have to write my own code.
    Note scipy.stats.uniform takes parameters loc and scale where the boundaries are defined to be
    loc and loc + scale, so scale = upper - lower bound.
    Returns list of fitted prior objects length of nDims (one function for each parameter)
    """
    priorFuncs = []
    priorFuncsPpf = []
    priorFuncsLogPdf = []
    priorType = priorParams[0, :]
    param1Vec = priorParams[1, :]
    param2Vec = priorParams[2, :]
    for i in range(len(priorType)):
        if priorType[i] == 1:
            priorFunc = scipy.stats.uniform(param1Vec[i],
                                            param2Vec[i] - param1Vec[i])
        elif priorType[i] == 2:
            priorFunc = scipy.stats.norm(param1Vec[i], param2Vec[i])
        else:
            print(
                "priors other than uniform and Gaussian not currently supported"
            )
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
        livePointsPhys[:, i] = priorFuncsPpf[i](livePoints[:, i])
    return livePointsPhys


def priorFuncsProd(livePoint, priorFuncsPdf):
    """
    calculates pdf of prior for each parameter dimension, then multiplies these together to get the pdf of the prior (i.e. the prior pdf assuming the parameters are independent)
    Works in linear space, but can easily be adapted if this ever leads to underflow errors (consider sum of log(pi(theta))).
    """
    livePointPriorValues = np.zeros_like(livePoint)
    for i in range(len(priorFuncsPdf)):
        livePointPriorValues[i] = priorFuncsPdf[i](livePoint[i])
    priorProdValue = livePointPriorValues.prod()
    return priorProdValue


def fitLhood(LLhoodParams):
    """
    fit lhood (without data) for parameters to make future evaluations much faster.
    """
    LLhoodType = LLhoodParams[0]
    mu = LLhoodParams[1].reshape(-1)
    sigma = LLhoodParams[2]
    if LLhoodType == 2:
        LhoodObj = scipy.stats.multivariate_normal(mu, sigma)
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


# Lhood sampling functions


def getNewLiveBlind(priorFuncsPpf, LhoodFunc, LhoodStar):
    """
    Blindly picks points isin U[0, 1]^D, converts these to physical values according to physical prior and uses as candidates for new livepoint until L > L* is found
    LhoodFunc can be Lhood or LLhood func, either works fine
    """
    trialPointLhood = -np.inf
    while trialPointLhood <= LhoodStar:
        nDims = len(priorFuncsPpf)
        trialPoint = np.random.rand(1, nDims)
        # Convert trialpoint value to physical value
        trialPointPhys = invPrior(trialPoint, priorFuncsPpf)
        # calculate LLhood value of trialpoint
        trialPointLhood = LhoodFunc(trialPointPhys)
    return trialPointPhys, trialPointLhood


def getNewLiveMH(livePointsPhys, deadIndex, priorFuncsPdf, priorParams,
                 LhoodFunc, LhoodStar):
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
    """
    nAccept = 0
    maxTrials = 80
    # current deadpoint not a possible starting candidate. This could be
    # ignored
    startCandidates = np.delete(livePointsPhys, (deadIndex), axis=0)
    trialSigma = calcInitTrialSig(livePointsPhys)
    # ensure that at least one move was made from initially picked point, or
    # new returned livepoint will be same as pre-existing livepoint.
    while nAccept == 0:
        # in the case of no acceptances, process is started again from step of picking starting livepoint
        # randomly pick starting candidate
        startIndex = np.random.randint(0, len(startCandidates[:, 0]))
        startPoint = startCandidates[startIndex]
        # this is not used per sae, but an arbitrary value is needed for first
        # value in loop
        startLhood = LhoodFunc(startPoint)
        nTrials = 0
        nReject = 0
        while nTrials < maxTrials:
            nTrials += 1
            # find physical values of trial point candidate
            trialPoint = np.random.multivariate_normal(startPoint,
                                                       trialSigma**2)
            # check trial point has physical values within sampling space
            # domain
            trialPoint = checkBoundaries(trialPoint, priorParams)
            trialLhood = LhoodFunc(trialPoint)
            # returns previous point values if test fails, or trial point
            # values if it passes
            acceptFlag, startPoint, startLhood = testTrial(
                trialPoint, startPoint, trialLhood, startLhood, LhoodStar,
                priorFuncsPdf)
            if acceptFlag:
                nAccept += 1
            else:
                nReject += 1
            # update trial distribution variance
            trialSigma = updateTrialSigma(trialSigma, nAccept, nReject)
    return startPoint, startLhood


# MH related functions


def calcInitTrialSig(livePoints):
    """
    calculate initial standard deviation based on width of domain defined by max and min parameter values of livepoints in each dimension
    """
    minParams = livePoints.min(axis=0)
    maxParams = livePoints.max(axis=0)
    livePointsWidth = maxParams - minParams
    # Sivia 2006 uses 0.1 * domain width
    trialSigma = np.diag(0.1 * livePointsWidth)
    return trialSigma


def testTrial(trialPoint, startPoint, trialLhood, startLhood, LhoodStar,
              priorFuncsPdf):
    """
    Check if trial point has L > L* and accept with probability prior(trial) / prior(previous)
    """
    newPoint = startPoint
    newLhood = startLhood
    acceptFlag = False
    if trialLhood > LhoodStar:
        prob = np.random.rand()
        priorRatio = priorFuncsProd(trialPoint,
                                    priorFuncsPdf) / priorFuncsProd(
                                        startPoint, priorFuncsPdf)
        if priorRatio > prob:
            newPoint = trialPoint
            newLhood = trialLhood
            acceptFlag = True
    return acceptFlag, newPoint, newLhood


def updateTrialSigma(trialSigma, nAccept, nReject):
    """
    update standard deviation as in Sivia 2006.
    Apparently this ensures that ~50% of the points are accepted, but I'm sceptical.
    """
    if nAccept > nReject:
        trialSigma = trialSigma * np.exp(1. / nAccept)
    else:
        trialSigma = trialSigma * np.exp(-1. / nReject)
    return trialSigma


def checkBoundaries(livePoint, priorParams):
    """
    For all parameters with fixed boundaries, (just uniform for NOW), ensures trial point in that dimension has value in allowed domain
    """
    # it makes most sense for these two to be the same
    differenceCorrection = 'reflect'
    pointCorrection = 'reflect'
    priorType = priorParams[0, :]
    param1Vec = priorParams[1, :]
    param2Vec = priorParams[2, :]
    for i in range(len(priorType)):
        if priorType[i] == 1:
            livePoint[i] = applyBoundary(livePoint[i], param1Vec[i],
                                         param2Vec[i], differenceCorrection,
                                         pointCorrection)
    return livePoint


def applyBoundary(point, lower, upper, differenceCorrection, pointCorrection):
    """
    give a point in or outside the domain, in case of point being in the domain it does nothing.
    When it is outside, it calculates a 'distance' from the domain according to differenceCorrection type, and then uses this 'distance' to transform the point into the domain by using either reflective or wrapping methods according to the value of pointCorrection.
    It is recommended that differenceCorrection and pointCorrection take the same values (makes most intuitive sense to me)
    """
    # get 'distance' from boundary
    if differenceCorrection == 'wrap':
        pointTemp = modToDomainWrap(point, lower, upper)
    elif differenceCorrection == 'reflect':
        pointTemp = modToDomainReflect(point, lower, upper)
    # use 'distance' to reflect or wrap point into domain
    if pointCorrection == 'wrap':
        if point < lower:
            point = upper - pointTemp
        elif point > upper:
            point = lower + pointTemp
    if pointCorrection == 'reflect':
        if point < lower:
            point = lower + pointTemp
        elif point > upper:
            point = upper - pointTemp
    return point


# following ensures point isin [lower, upper]. This wraps according to
# (positive) difference between point and nearest bound and returns a
# 'distance' which can actually be used to get the correct value of the
# point within the domain. Effect of this is basically mod'ing the
# difference by the width of the domain. It makes most intuitive sense to
# me to use this when you want to wrap the points around the domain.
def modToDomainWrap(point, lower, upper):
    return (lower - point) % (upper - lower) if point < lower else (
        point - upper) % (upper - lower)


def modToDomainReflect(point, lower, upper):
    """
    following ensures point isin [lower, upper]. This reflects according to (positive) difference between point and nearest bound and returns a 'distance' which can actually be used to get the correct value of the point within the domain. It makes most intuitive sense to me to use this when you want to reflect the points in the domain.
    Operation done to ensure reflecting is different based on whether the difference between the point and the nearest part of the domain is an odd or even (incl. 0) multiple of the boundary.
    """
    if point < lower:
        # number of multiples (truncated) of the width of the domain the point
        # lays outside it
        outsideMultiple = (lower - point) // (upper - lower)
        # checks if number of multiples of width of domain the point is outside
        # the domain is odd or even (the latter including zero)
        oddFlag = outsideMultiple % 2
        if oddFlag:
            # in this case for a reflective value the mod'd distance needs to
            # be counted from the opposite boundary. This can be done by
            # calculating - delta mod width where delta is difference between
            # closest boundary and point
            pointTemp = (point - lower) % (upper - lower)
        else:
            # this is the simpler case in which the reflection is counted from
            # the nearest boundary which is just delta mod width
            pointTemp = (lower - point) % (upper - lower)
    elif point > upper:
        # as above but delta is calculated from upper bound
        outsideMultiple = (point - upper) // (upper - lower)
        oddFlag = outsideMultiple % 2  # as above
        if oddFlag:
            # as above
            pointTemp = (upper - point) % (upper - lower)
        else:
            # as above
            pointTemp = (point - upper) % (upper - lower)
    else:
        pointTemp = None
    return pointTemp


# setup related functions


def checkInputParamsShape(priorParams, LhoodParams, nDims):
    """
    checks prior and Lhood input arrays are correct shape
    """
    assert (priorParams.shape == (
        3, nDims)), "Prior parameter array should have shape (3, nDims)"
    assert (LhoodParams[1].shape == (
        1, nDims)), "Llhood params mean array should have shape (1, nDims)"
    assert (LhoodParams[2].shape == (
        nDims,
        nDims)), "LLhood covariance array should have shape (nDims, nDims)"


# NS loop related functions


def tryTerminationLog(verbose, terminationType, terminationFactor, nest, nLive,
                      logEofX, livePointsLLhood, LLhoodStar, ZLiveType,
                      trapezoidalFlag, logEofZ, H):
    """
    See if termination condition for main loop of NS has been met. Can be related to information value H or whether estimated remaining evidence is below a given fraction of the Z value calculated up to that iteration
    """
    breakFlag = False
    if terminationType == 'information':
        terminator = terminationFactor * nLive * H
        if verbose:
            printTerminationUpdateInfo(nest, terminator)
        if nest > terminator:
            # since it is terminating need to calculate remaining Z
            liveMaxIndex, liveLLhoodMax, logEofZLive, avLLhood, nFinal = getLogEofZLive(
                nLive, logEofX, livePointsLLhood, LLhoodStar, ZLiveType,
                trapezoidalFlag)
            breakFlag = True
        else:
            liveMaxIndex = None  # no point calculating
            liveLLhoodMax = None  # these values if not terminating
    elif terminationType == 'evidence':
        liveMaxIndex, liveLLhoodMax, logEofZLive, avLLhood, nFinal = getLogEofZLive(
            nLive, logEofX, livePointsLLhood, LLhoodStar, ZLiveType,
            trapezoidalFlag)
        endValue = np.exp(logEofZLive - logEofZ)
        if verbose:
            printTerminationUpdateZ(logEofZLive, endValue, terminationFactor,
                                    'log')
        if endValue <= terminationFactor:
            breakFlag = True
    return breakFlag, liveMaxIndex, liveLLhoodMax, avLLhood, nFinal


def tryTermination(verbose, terminationType, terminationFactor, nest, nLive,
                   EofX, livePointsLhood, LhoodStar, ZLiveType,
                   trapezoidalFlag, EofZ, H):
    """
    as above but in linear space
    """
    breakFlag = False
    if terminationType == 'information':
        terminator = terminationFactor * nLive * H
        if verbose:
            printTerminationUpdateInfo(nest, terminator)
        if nest > terminator:
            liveMaxIndex, liveLhoodMax, ZLive, avLhood, nFinal = getEofZLive(
                nLive, EofX, livePointsLhood, LhoodStar, ZLiveType,
                trapezoidalFlag)
            breakFlag = True
        else:
            liveMaxIndex = None
            liveLhoodMax = None
    elif terminationType == 'evidence':
        liveMaxIndex, liveLhoodMax, EofZLive, avLhood, nFinal = getEofZLive(
            nLive, EofX, livePointsLhood, LhoodStar, ZLiveType,
            trapezoidalFlag)
        endValue = EofZLive / EofZ
        if verbose:
            printTerminationUpdateZ(EofZLive, endValue, terminationFactor,
                                    'linear')
        if endValue <= terminationFactor:
            breakFlag = True
    return breakFlag, liveMaxIndex, liveLhoodMax, avLhood, nFinal


def getLogEofZLive(nLive, logEofX, livePointsLLhood, LLhoodStar, ZLiveType,
                   trapezoidalFlag):
    """
    NOTE logWeightsLive here is an np array
    newLiveLLhoods has same shape as logWeightsLive (i.e. account for averageLhoodOrX value). If ZLiveType == 'max' avLLhood will just be the maximum LLhood value.
    there is no averaging to consider if ZLiveType == 'max Lhood'.
    Could return live weights, but these need to be calculated again in final contribution function so don't bother
    """
    livePointsLLhood2, liveLLhoodMax, liveMaxIndex = getMaxLhood(
        ZLiveType, livePointsLLhood)
    logEofwLive = getLogEofwLive(nLive, logEofX, ZLiveType)
    # this will be an array nLive long for 'average' ZLiveType and 'X'
    # averageLhoodOrX or a 1 element array for 'max' ZLiveType or 'Lhood'
    # averageLhoodOrX
    logEofWeightsLive, avLLhood, nFinal = getLogEofWeightsLive(
        logEofwLive, LLhoodStar, livePointsLLhood2, trapezoidalFlag, ZLiveType)
    logEofZLive = logAddArr2(-np.inf, logEofWeightsLive)
    return liveMaxIndex, liveLLhoodMax, logEofZLive, avLLhood, nFinal


def getEofZLive(nLive, EofX, livePointsLhood, LhoodStar, ZLiveType,
                trapezoidalFlag):
    """
    as above but in linear space
    """
    livePointsLhood2, liveLhoodMax, liveMaxIndex = getMaxLhood(
        ZLiveType, livePointsLhood)
    EofwLive = getEofwLive(nLive, EofX, ZLiveType)
    EofWeightsLive, avLhood, nFinal = getEofWeightsLive(
        EofwLive, LhoodStar, livePointsLhood2, trapezoidalFlag, ZLiveType)
    EofZLive = np.sum(EofWeightsLive)
    return liveMaxIndex, liveLhoodMax, EofZLive, avLhood, nFinal


def getMaxLhood(ZLiveType, livePointsLhood):
    """
    For ZLiveType == 'max' returns a 1 element array with maximum LLhood value, its value as a scalar, and the index of the max LLhood in the given array.
    For ZLiveType == 'average' it essentially does nothing
    """
    if 'average' in ZLiveType:  # average of remaining LLhood values/ X for final Z estimate
        livePointsLhood2 = livePointsLhood
        liveLhoodMax = None  # liveLLhoodMax is redundant for this method so just return None for it
        liveMaxIndex = None  # same as line above
    elif ZLiveType == 'max Lhood':  # max of remaining LLhood values & remaining X for final Z estimate
        liveMaxIndex = np.argmax(livePointsLhood)
        liveLhoodMax = np.asscalar(livePointsLhood[liveMaxIndex])
        livePointsLhood2 = np.array([liveLhoodMax])
    return livePointsLhood2, liveLhoodMax, liveMaxIndex


def getLogEofwLive(nLive, logEofX, ZLiveType):
    """
    Determines final logw based on ZLiveType and averageLhoodOrX, i.e. it determines whether final contribution is averaged/ maximised over L or averaged over X.
    """
    if (ZLiveType == 'max Lhood') or (ZLiveType == 'average Lhood'):
        return logEofX
    else:
        return logEofX - np.log(nLive)


def getEofwLive(nLive, EofX, ZLiveType):
    """
    as above but in non-log space
    """
    if (ZLiveType == 'max Lhood') or (ZLiveType == 'average Lhood'):
        return EofX
    else:
        return EofX / nLive


def getLogEofWeightsLive(logEofw, LLhoodStar, liveLLhoods, trapezoidalFlag,
                         ZLiveType):
    """
    From Will's implementation, Z = sum (X_im1 - X_i) * 0.5 * (L_i + L_im1)
    Unsure whether you should treat final contribution using trapezium rule (when it is used for rest of sum). I think you should
    and in case of ZLiveType == 'average *', the L values used are L* + {L_live}
    and in the case of ZLiveType == 'max', the L values used are L* + {max(L_live)}.
    When trapezium rule isn't used (for rest of sum), L values used are
    {L_live} in case of ZLiveType == 'average *'
    and {max(L_live)} in case of ZLiveType == 'max'.
    When ZLiveType == 'average *' there is an added complication of what the average is 'taken over' (for both trapezium rule and standard quadrature) i.e. over the prior volume or the likelihood.
    If ZLiveType == 'average X' the average is taken over X, meaning there are still nLive live log weights (equally spaced in X with values X / nLive) which for standard quadrature have values: {log(X / nLive) + log(L_1), ..., log(X / nLive) + log(L_nLive)}
    and for trapezium rule: {log(X / nLive) + log((L* + L_1) / 2. ), ..., log(X / nLive) + log((L_nLive-1 + L_nLive) / 2. )}
    If ZLiveType == 'average Lhood' the average is taken over the remaining L values, meaning there is 1 live log weight with X value X (i.e. the L_average value is assumed to be at X = 0). For the standard quadrature method the live log weight thus has a value log(X) + log(sum_i^nLive[L_i] / nLive)
    and for the trapezoidal rule log(X) + log((L* + sum_i^nLive[L_i] / nLive) / 2.).
    When ZLiveType == 'max', the maximum is obviously taken over the remaining Lhoods. Thus there is only one live log weight. For standard quadrature this is log(X) + log(max(L_i)
    and for the trapezium rule it is log(X) + log((L* + max(L_i)) / 2.)
    If averaging over L, final livepoint needs to be attributed this L, so it is stored here under the variable avLLhood
    """
    if trapezoidalFlag:
        if ZLiveType == 'average X':  # assumes there is still another nLive points to be added to the posterior samples, as averaging is done over X, not L
            nFinal = len(liveLLhoods)
            # slower than appending lists together, but liveLLhoods is a numpy
            # array, and converting it to a list is slow
            laggedLLhoods = np.concatenate(([LLhoodStar], liveLLhoods[:-1]))
            logEofWeightsLive = logEofw + \
                np.log(0.5) + np.logaddexp(liveLLhoods, laggedLLhoods)
            avLLhood = None  # if not averaging over Lhood this isn't needed
        else:  # assumes 'final' Lhood value is given by the average of the remaining L values, and that this is at X = 0
            nFinal = 1
            # Make array for consistency
            LSumLhood = np.array([logAddArr2(-np.inf, liveLLhoods)])
            # 1 for ZLiveType == 'max' or nLive for ZLiveType == 'average
            # Lhood'
            n = len(liveLLhoods)
            avLLhood = LSumLhood - np.log(n)
            logEofWeightsLive = np.log(0.5) + logEofw + np.logaddexp(
                LLhoodStar, avLLhood)
    else:
        if ZLiveType == 'average X':
            nFinal = len(liveLLhoods)
            logEofWeightsLive = logEofw + liveLLhoods
            avLLhood = None
        else:
            nFinal = 1
            LSumLhood = np.array([logAddArr2(-np.inf, liveLLhoods)])
            n = len(liveLLhoods)
            avLLhood = LSumLhood - np.log(n)
            logEofWeightsLive = logEofw + avLLhood
    return logEofWeightsLive, avLLhood, nFinal


def getEofWeightsLive(Eofw, LhoodStar, liveLhoods, trapezoidalFlag, ZLiveType):
    """
    as above but non-log space version
    """
    if trapezoidalFlag:
        if ZLiveType == 'average X':
            nFinal = len(liveLhoods)
            laggedLhoods = np.concatenate(([LhoodStar], liveLhoods[:-1]))
            EofWeightsLive = Eofw * 0.5 * (liveLhoods + laggedLhoods)
            avLhood = None
        else:
            nFinal = 1
            sumLhood = np.array([liveLhoods.sum()])
            n = len(liveLhoods)
            avLhood = sumLhood / n
            EofWeightsLive = Eofw * 0.5 * (LhoodStar + avLhood)
    else:
        if ZLiveType == 'average X':
            nFinal = len(liveLhoods)
            EofWeightsLive = Eofw * liveLhoods
            avLhood = None
        else:
            nFinal = 1
            sumLhood = np.array([liveLhoods.sum()])
            n = len(liveLhoods)
            avLhood = sumLhood / n
            EofWeightsLive = Eofw * avLhood
    return EofWeightsLive, avLhood, nFinal


# final contribution to NS sampling functions


def getFinalContributionLog(verbose,
                            ZLiveType,
                            trapezoidalFlag,
                            nFinal,
                            logEofZ,
                            logEofZ2,
                            logEofX,
                            logEofWeights,
                            H,
                            livePointsPhys,
                            livePointsLLhood,
                            avLLhood,
                            liveLLhoodMax,
                            liveMaxIndex,
                            LLhoodStar,
                            errorEval='recursive'):
    """
    Get final contribution from livepoints after NS loop has ended. Way of estimating final contribution is dictated by ZLiveType.
    Also updates H value and gets final weights (and physical values) for posterior
    this function could be quite taxing on memory as it has to copy all arrays/ lists across
    NOTE: for standard quadrature summation, average Lhood and average X give same values of Z (averaging over X is equivalent to averaging over L). However, correct posterior weights are given by latter method, and Z errors are different in both cases
    """
    livePointsLLhood = checkIfAveragedLhood(
        nFinal, livePointsLLhood,
        avLLhood)  # only relevant for 'average' ZLiveType
    if 'average' in ZLiveType:
        LLhoodsFinal = np.concatenate(
            (np.array([LLhoodStar]), livePointsLLhood))
        logEofZOld = logEofZ
        logEofZ2Old = logEofZ2
        for i in range(
                nFinal
        ):  # add weight of each remaining live point incrementally so H can be calculated easily (according to formulation given in Skilling)
            logEofZLive, logEofZ2Live, logEofWeightLive = updateLogZnXMomentsFinal(
                nFinal, logEofZOld, logEofZ2Old, logEofX, LLhoodsFinal[i],
                LLhoodsFinal[i + 1], trapezoidalFlag, errorEval)
            logEofWeights.append(logEofWeightLive)
            H = updateHLog(H, logEofWeightLive, logEofZLive,
                           LLhoodsFinal[i + 1], logEofZOld)
            logEofZOld = logEofZLive
            logEofZ2Old = logEofZ2Live
            if verbose:
                printFinalLivePoints(i, livePointsPhys[i], LLhoodsFinal[i + 1],
                                     ZLiveType, 'log')
        livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr = getFinalAverage(
            livePointsPhys, livePointsLLhood, logEofX, nFinal, avLLhood, 'log')
    # assigns all remaining prior mass to one point which has highest
    # likelihood (of remaining livepoints)
    elif ZLiveType == 'max Lhood':
        logEofZLive, logEofZ2Live, logEofWeightLive = updateLogZnXMomentsFinal(
            nFinal, logEofZ, logEofZ2, logEofX, LLhoodStar, liveLLhoodMax,
            trapezoidalFlag, errorEval)
        # add scalar to list (as in 'average' case) instead of 1 element array
        logEofWeights.append(logEofWeightLive)
        H = updateHLog(H, logEofWeightLive, logEofZLive, liveLLhoodMax,
                       logEofZ)
        livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr = getFinalMax(
            liveMaxIndex, livePointsPhys, liveLLhoodMax, logEofX)
        if verbose:
            printFinalLivePoints(liveMaxIndex, livePointsPhysFinal,
                                 livePointsLLhoodFinal, ZLiveType, 'log')
    return logEofZLive, logEofZ2Live, H, livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr


def getFinalContribution(verbose,
                         ZLiveType,
                         trapezoidalFlag,
                         nFinal,
                         EofZ,
                         EofZ2,
                         EofX,
                         EofWeights,
                         H,
                         livePointsPhys,
                         livePointsLhood,
                         avLhood,
                         liveLhoodMax,
                         liveMaxIndex,
                         LhoodStar,
                         errorEval='recursive'):
    """
    as above but in linear space
    """
    livePointsLhood = checkIfAveragedLhood(nFinal, livePointsLhood, avLhood)
    if (ZLiveType == 'average Lhood') or (ZLiveType == 'average X'):
        EofZOld = EofZ
        EofZ2Old = EofZ2
        LhoodsFinal = np.concatenate((np.array([LhoodStar]), livePointsLhood))
        for i in range(nFinal):
            EofZLive, EofZ2Live, EofWeightLive = updateZnXMomentsFinal(
                nFinal, EofZOld, EofZ2Old, EofX, LhoodsFinal[i],
                LhoodsFinal[i + 1], trapezoidalFlag, 'recursive')
            EofWeights.append(EofWeightLive)
            H = updateH(H, EofWeightLive, EofZLive, LhoodsFinal[i + 1],
                        EofZOld)
            EofZOld = EofZLive
            EofZ2Old = EofZ2Live
            if verbose:
                printFinalLivePoints(i, livePointsPhys[i], LhoodsFinal[i + 1],
                                     ZLiveType, 'linear')
        livePointsPhysFinal, livePointsLhoodFinal, EofXFinalArr = getFinalAverage(
            livePointsPhys, livePointsLhood, EofX, nFinal, avLhood, 'linear')
    elif ZLiveType == 'max Lhood':
        EofZLive, EofZ2Live, EofWeightLive = updateZnXMomentsFinal(
            nFinal, EofZ, EofZ2, EofX, LhoodStar, liveLhoodMax,
            trapezoidalFlag, 'recursive')
        EofWeights.append(EofWeightLive)
        H = updateH(H, EofWeightLive, EofZLive, liveLhoodMax, EofZ)
        livePointsPhysFinal, livePointsLhoodFinal, EofXFinalArr = getFinalMax(
            liveMaxIndex, livePointsPhys, liveLhoodMax, EofX)
        if verbose:
            printFinalLivePoints(liveMaxIndex, livePointsPhysFinal,
                                 livePointsLhoodFinal, ZLiveType, 'linear')
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


def getFinalAverage(livePointsPhys, livePointsLLhood, X, nFinal, avLLhood,
                    space):
    """
    gets final livepoint values and X value per remaining livepoint for average Z criteria
    NOTE Xfinal is a list not a numpy array
    space says whether you are working in linear or log space (X or logX)
    """
    livePointsPhysFinal = getLivePointsPhysFinal(
        livePointsPhys, avLLhood)  # only relevant for 'average' ZLiveType
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
    if not avLhood:  # ZLiveType == 'max' means livePointsPhys is already just one livepoint, averageLhoodOrX == 'average X' means retain previous array
        return livePointsPhys
    # need to obtain one livepoint from set of nLive. NO (KNOWN AT STAGE OF
    # ALGORITHM) PHYSICAL VECTOR CORRESPONDS TO THIS LIKEILIHOO, SO THIS VALUE
    # IS MEANINGLESS
    else:
        return livePointsPhys.mean(axis=0).reshape(1, -1)


def getFinalMax(liveMaxIndex, livePointsPhys, liveLhoodMax, X):
    """
    get livepoint and physical livepoint values
    corresponding to maximum likelihood point in remaining
    livepoints.
    Note Xfinal is a list not a numpy array or a scalar
    Function works for log or linear space
    """
    livePointsPhysFinal = livePointsPhys[liveMaxIndex].reshape(1, -1)
    # for consistency with 'average' equivalent function
    livePointsLhoodFinal = np.array([liveLhoodMax])
    # for consistency with 'average' equivalent function, make it a list.
    Xfinal = [X]
    return livePointsPhysFinal, livePointsLhoodFinal, Xfinal


# final datastructure / output functions


def getTotal(deadPointsPhys, livePointsPhysFinal, deadPointsLhood,
             livePointsLhoodFinal, XArr, XFinalArr, weights):
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


def writeOutput(outputFile, totalPointsPhys, totalPointsLhood, weights, XArr,
                paramNames, space):
    """
    writes a summary file which contains values for all sampled points.
    Also writes files needed for getDist.
    """
    paramNamesStr = ', '.join(paramNames)
    if space == 'linear':
        summaryStr = ' Lhood, weights, X'
    else:
        summaryStr = ' LLhood, logWeights, logX'
    # summary file containing most information of sampled points
    np.savetxt(outputFile + '_summary.txt',
               np.column_stack(
                   (totalPointsPhys, totalPointsLhood, weights, XArr)),
               delimiter=',',
               header=paramNamesStr + summaryStr)
    # chains file in format needed for getDist: importance weight (weights or
    # logWeights), LHood (Lhood or LLhood), phys param values
    np.savetxt(outputFile + '.txt',
               np.column_stack((weights, totalPointsLhood, totalPointsPhys)))
    # index and list parameter names for getDist
    nameFile = open(outputFile + '.paramnames', 'w')
    for i, name in enumerate(paramNames):
        nameFile.write('p%i %s\n' % (i + 1, name))
    nameFile.close()
    # write file with hard constraints on parameter boundaries.
    # Hard constraints are currently inferred from data for all parameters
    rangeFile = open(outputFile + '.ranges', 'w')
    for i in range(len(paramNames)):
        rangeFile.write('p%i N N\n' % (i + 1))
    rangeFile.close()


# Z & H theoretical functions


def nDIntegratorZTheor(integrandFuncs, limitsList, integrandLogVal=200.):
    """
    integrator used for calculating theoretical value of Z. integrand is function which evaluates to value of integrand at given parameter values (which are determined by nquad function).
    integrandFuncs is list of functions (Lhood & non-rectangular priors) which are multiplied together to give value of integrand.
    If integrandLogVal evaluates to true, does integration method which takes exp of log of integrand * some large number given by integrandLogVal
    to avoid underflow. The final integral result (and error) is then divided by exp(integrandLogVal) to get the final value.
    """
    if integrandLogVal:
        return scipy.integrate.nquad(
            evalExpLogIntegrand,
            limitsList,
            args=(integrandFuncs, integrandLogVal)) / np.exp(integrandLogVal)
    else:  # this should only occur if you aren't concerned about underflow
        return scipy.integrate.nquad(evalIntegrand,
                                     limitsList,
                                     args=(integrandFuncs, ))


def nDIntegratorHTheor(integrandFuncs,
                       limitsList,
                       integrandLogVal=200.,
                       LLhoodFunc=None):
    """
    As above but has to call nquad with a  slightly different function representing integrand when using
    exp(log(integrand)) method for calculating, due to LLhood(theta) part of integrand
    """
    if integrandLogVal:
        return scipy.integrate.nquad(
            evalLogLExpLogIntegrand,
            limitsList,
            args=(integrandFuncs, integrandLogVal,
                  LLhoodFunc)) / np.exp(integrandLogVal)
    else:  # this should only occur if you aren't concerned about underflow
        return scipy.integrate.nquad(evalIntegrand,
                                     limitsList,
                                     args=(integrandFuncs, ))


def evalIntegrand(*args):
    """
    evaluates a-priori parameter fitted pdfs with given data. data has to be reshaped because scipy functions are annoying and require last axis to be number of dimensions.
    Last element of parametersAndIntegrandFuncs is list of pdf objects for Lhood and priors, which are evaluated and multiplied together to give the value of the integrand.
    Note in general, Lhood will have dimensionality nDims whereas each prior will have dimensionality 1.
    The value (of the key corresponding to the function) is the relevant slice of x to get the correct dimensions
    for each function.
    May suffer from underflow either when evaluating the .pdf() calls, or when multiplying them together.
    args consists of 1) the vector of theta values for given call (from nquad)
    2) list of functions which make up the integrand
    """
    theta = np.array(args[:-1]).reshape(1, -1)
    integrandFuncs = args[-1]
    integrandVal = 1.
    for func, argIndices in integrandFuncs.items():
        integrandVal *= func(theta[argIndices])
    return integrandVal


def evalExpLogIntegrand(*args):
    """
    evaluates .logpdf() of Lhood/ prior functions, adds them together
    along with an arbitrary 'large' value given in paramsNIntLogFuncsNIntLogVal[-1]
    then exponentiates this value to avoid underflow when evaluating the pdfs/ multiplying them together.
    This should avoid any underflow at all, provided the given value of integrandLogVal is large enough
    There is some underflow you cannot avoid, regardless of the value of integrandLogVal. This is because the LLhood values range from e.g. O(-10) to ~ O(-10^3).
    Subtracting approximation of max value of integrand is also not helpful for the same reason (spans too many orders of magnitude)
    Using an integrandLogVal suitable for the former will still cause the latter to underflow,
    but using a value to prevent the latter from underflow will result in the former overflowing!
    Dynamically calculating integrandLogVal wouldn't help, as you would need to include this number again after exponentiating (which would cause it to underflow), but before the value is added to integral (if factor isn't constant, doesn't commute with integral)
    n.b. underflow is usually better than overflow, as you don't want to miss your most likely values
    args consists of 1) the vector of theta values for given call (from nquad)
    2) list of functions which make up the integrand
    3) 'large' number to be added to logarithm before exponentiated to give integrand
    """
    theta = np.array(args[:-2]).reshape(1, -1)
    integrandLogFuncs = args[-2]
    integrandLogVal = args[-1]
    for logFunc, argIndices in integrandLogFuncs.items():
        integrandLogVal += logFunc(theta[argIndices])
    return np.exp(integrandLogVal)


def evalLogLExpLogIntegrand(*args):
    """
    Same as above evalExpLogIntegrand but multiplies by log(L) for calculating H
    using exp(log(integrand)) method.
    LLhoodFunc has to be passed in separately as well as in the dictionary of functions, so it can be used
    to evaluate LLhoodFunc(theta)*np.exp(integrandLogVal)
    args consists of 1) the vector of theta values for given call (from nquad)
    2) list of functions which make up the integrand
    3) 'large' number to be added to logarithm before exponentiated to give integrand
    4) LLhood function required for log(L(theta)) part of integrand
    """
    theta = np.array(args[:-3]).reshape(1, -1)
    integrandLogFuncs = args[-3]
    integrandLogVal = args[-2]
    LLhoodFunc = args[-1]
    LLhoodFuncArgs = integrandLogFuncs[LLhoodFunc]  # should be all of theta
    for logFunc, argIndices in integrandLogFuncs.items():
        integrandLogVal += logFunc(theta[argIndices])
    return LLhoodFunc(theta[LLhoodFuncArgs]) * np.exp(integrandLogVal)


def getPriorIntegrandAndLimits(priorParams, priorFuncsPdf, integrandFuncs):
    """
    Adds prior functions to integrandFuncs dict for 'non-rectangular' (uniform) dimensions, as long as a mapping to dimensions of data required to integrate over for that function. For uniform priors, calculates the hyperrectangular volume. Also creates list of limits in each dimension of integral. For Gaussian priors, sets limits to +- infinity
    """
    hyperRectangleVolume = 1.
    priorTypes = priorParams[0, :]
    bounds = priorParams[1:, :]
    limitsList = []
    for i in range(len(priorTypes)):
        if priorTypes[i] == 1:  # uniform
            limitsList.append(np.array([priorParams[1, i], priorParams[2, i]]))
            priorWidth = priorParams[2, i] - priorParams[1, i]
            hyperRectangleVolume *= priorWidth
        elif priorTypes[i] == 2:  # gauss
            # parameters of Gauss dists aren't limits, so change to +-inf here
            limitsList.append(np.array([-np.inf, np.inf]))
            # tuple representing slice of data array x for this prior function,
            # is mapped to that prior function via the dictionary
            integrandFuncs[priorFuncsPdf[i]] = (0, i)
    return integrandFuncs, limitsList, hyperRectangleVolume


class ZTheorException(Exception):
    pass


def calcZTheor(priorParams, priorFuncsPdf, LhoodFunc, nDims):
    """
    numerically integrates L(theta) * pi(theta) over theta
    priorFuncs must be in same order as dimensions of LhoodFunc when it was fitted (and in same order as priorParams).
    Have to temporarily set np.seterr to warnings only as underflow always seems to occur when evaluating integral (make sense I guess).
    A bit slow, but not sure how I can make it faster tbh, as it will get exponentially slower with # of dimensions
    LhoodFunc and priorFuncsPdf can be .pdf() or .logpdf() methods
    """
    np.seterr(all='warn')
    # slice refers to which dimensions of data array are required for given
    # function in integration call. In case of Lhood, all dimensions of the
    # parameter space are required
    integrandFuncs = {LhoodFunc: slice(None)}
    integrandFuncs, limitsList, hyperRectangleVolume = getPriorIntegrandAndLimits(
        priorParams, priorFuncsPdf, integrandFuncs)
    ZIntegral, ZIntegralE = nDIntegratorZTheor(integrandFuncs, limitsList)
    ZTheor = 1. / hyperRectangleVolume * ZIntegral
    np.seterr(all='raise')
    return ZTheor, ZIntegralE, hyperRectangleVolume


def calcZTheorApprox(priorParams):
    """
    Only valid in limit that prior is hyperrectangle, and majority of lhood is contained in prior hypervolume
    such that limits of integration (domain of the sampling space defined by the prior) can be extended close enough +- infinity such that the Lhood integrates to 1 over this domain
    """
    priorDifferences = priorParams[2, :] - priorParams[1, :]
    priorVolume = priorDifferences.prod()
    ZTheor = 1. / priorVolume
    return ZTheor, priorVolume


def calcHTheor(priorParams,
               priorFuncsPdf,
               LLhoodFunc,
               nDims,
               Z,
               ZErr,
               LhoodFunc=None):
    """
    Calculates HTheor from the KL divergence equation: H = int[P(theta) * ln(P(theta) / pi(theta))] = 1/Z * int[L(theta) * pi(theta) * ln(L(theta))] - ln(Z).
    For uniform priors, calculates volume and skips that part of integral (over pi(theta)).
    Uses same trick as ZTheor in that it composes a dictionary of functions to integrate, mapped to dimension(s) of theta vector to integrate along for given function.
    Passes this dictionary to function which nquad actually evaluates.
    """
    np.seterr(all='warn')
    if LhoodFunc:  # calculate H without considering underflow
        # slice refers to which dimensions of data array are required for given
        # function in integration call. In case of Lhood, all dimensions of the
        # parameter space are required
        integrandFuncs = {LhoodFunc: slice(None), LLhoodFunc: slice(None)}
        integrandLogVal = None
    else:
        # evaluate L(theta) using exp(log(L(theta)))
        integrandFuncs = {LLhoodFunc: slice(None)}
        integrandLogVal = 100.
    integrandFuncs, limitsList, hyperRectangleVolume = getPriorIntegrandAndLimits(
        priorParams, priorFuncsPdf, integrandFuncs)
    LhoodPiLogLIntegral, LhoodPiLogLErr = nDIntegratorHTheor(
        integrandFuncs, limitsList, integrandLogVal, LLhoodFunc)
    HErr = calcHErr(Z, ZErr, LhoodPiLogLIntegral, LhoodPiLogLErr)
    return 1. / (hyperRectangleVolume * Z) * \
        LhoodPiLogLIntegral - np.log(Z), HErr


def calcHErr(Z, ZErr, LhoodPiLogLIntegral, LhoodPiLogLErr):
    """
    Calculates error on H due to uncertainty of Z, HIntegrand and ln(Z).
    ignores possible correlation between Z and LhoodPiLogLIntegral
    """
    logZErr = ZErr / Z
    IntOverZErr = LhoodPiLogLIntegral / Z * \
        np.sqrt((ZErr / Z)**2. + (LhoodPiLogLErr / LhoodPiLogLIntegral)**2.)
    return np.sqrt(logZErr**2. + IntOverZErr**2.)


def calcHTheorApprox(Z, nDims, priorVolume):
    """
    Only valid in limit that prior is hyperrectangle, and majority of lhood is contained in prior hypervolume
    such that limits of integration can be extended to +- infinity
    """
    return -0.5 * nDims / (Z * priorVolume) * \
        (1. + np.log(2. * np.pi)) - np.log(Z)


# Updating expected values of Z, X and H functions


def calct(nLive, expectation='t', sampling=False, maxPoints=False):
    """
    calc value of t from its pdf,
    from (supposedely equivalent) way of deriving form of pdf,
    or from E[.] or E[l(.)]	"""
    if sampling:
        if maxPoints:
            t = np.random.rand(nLive).max()
        else:
            t = np.random.rand()**(1. / nLive)
    else:
        if expectation == 'logt':
            t = np.exp(-1. / nLive)
        elif expectation == 't':
            t = nLive / (nLive + 1.)
    return t


def calct2(nLive, expectation='t2', sampling=False, maxPoints=False):
    """
    calc value of t^2 from its pdf,
    from (supposedely equivalent) way of deriving form of pdf,
    or from E[.] or E[l(.)]
    """
    if sampling:
        if maxPoints:
            # TODO
            pass
        else:
            # TODO
            pass
    else:
        if expectation == 'logt2':
            # TODO
            pass
        elif expectation == 't2':
            t = nLive / (nLive + 2.)
    return t


def calc1mt(nLive, expectation='1mt', sampling=False, maxPoints=False):
    """
    calc value of 1-t from its pdf,
    from (supposedely equivalent) way of deriving form of pdf,
    or from E[.] or E[l(.)]
    """
    if sampling:
        if maxPoints:
            # TODO
            pass
        else:
            # TODO
            pass
    else:
        if expectation == 'log1mt':
            # TODO
            pass
        elif expectation == '1mt':
            t = 1. / (nLive + 1.)
    return t


def calc1mt2(nLive, expectation='1mt2', sampling=False, maxPoints=False):
    """
    calc value of (1-t)^2 from its pdf,
    from (supposedely equivalent) way of deriving form of pdf,
    or from E[.] or E[l(.)]
    """
    if sampling:
        if maxPoints:
            # TODO
            pass
        else:
            # TODO
            pass
    else:
        if expectation == 'log1mt2':
            # TODO
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


def updateZnXMoments(nLive, EofZ, EofZ2, EofZX, EofX, EofX2, LhoodStarOld,
                     LhoodStar, trapezoidalFlag):
    """
    Wrapper around updateZnXM taking into account whether trapezium rule is used or not
    """
    if trapezoidalFlag:
        EofZ, EofZ2, EofZX, EofX, EofX2, EofWeight = updateZnXM(
            nLive, EofZ, EofZ2, EofZX, EofX, EofX2,
            0.5 * (LhoodStarOld + LhoodStar))
    else:
        EofZ, EofZ2, EofZX, EofX, EofX2, EofWeight = updateZnXM(
            nLive, EofZ, EofZ2, EofZX, EofX, EofX2, LhoodStar)
    return EofZ, EofZ2, EofZX, EofX, EofX2, EofWeight


def updateZnXM(nLive, EofZ, EofZ2, EofZX, EofX, EofX2, L):
    """
    Update moments of Z and X based on their previous values, expected value of random variable t and Lhood value ((L_i + L_i-1) / 2. in case of trapezium rule).
    Used to calculate the mean and standard deviation of Z, and thus of log(Z) as well
    TODO: CONSIDER KEETON NON-RECURSIVE METHOD
    """
    Eoft, Eoft2, Eof1mt, Eof1mt2 = calcEofts(nLive)
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


def calcLogEofts(nLive):
    """
    Calculate log(E[t] - E[t^2]) as it is much easier to do so here than later having on log(E[t]) and log(E[t^2])
    """
    return np.log(calcEofts(nLive) + (calct(nLive) - calct2(nLive), ))


def updateLogZnXMoments(nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2,
                        LLhoodStarOld, LLhoodStar, trapezoidalFlag):
    """
    as above but for log space
    """
    if trapezoidalFlag:
        logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight = updateLogZnXM(
            nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2,
            np.log(0.5) + np.logaddexp(LLhoodStarOld, LLhoodStar))
    else:
        logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight = updateLogZnXM(
            nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, LLhoodStar)
    return logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight


def updateLogZnXM(nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2, LL):
    """
    as above but for log space
    """
    logEoft, logEoft2, logEof1mt, logEof1mt2, logEoftmEoft2 = calcLogEofts(
        nLive)
    logEofZ, logEofWeight = updateLogEofZ(logEofZ, logEof1mt, logEofX, LL)
    logEofZ2 = updateLogEofZ2(logEofZ2, logEof1mt, logEofZX, logEof1mt2,
                              logEofX2, LL)
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


def updateZnXMomentsFinal(nFinal, EofZ, EofZ2, EofX, Lhood_im1, Lhood_i,
                          trapezoidalFlag, errorEval):
    """
    Wrapper around updateZnXMomentsF taking into account whether trapezium rule is used or not
    """
    if trapezoidalFlag:
        EofZ, EofZ2, EofWeight = updateZnXMomentsF(nFinal, EofZ, EofZ2, EofX,
                                                   (Lhood_im1 + Lhood_i) / 2.,
                                                   errorEval)
    else:
        EofZ, EofZ2, EofWeight = updateZnXMomentsF(nFinal, EofZ, EofZ2, EofX,
                                                   Lhood_i, errorEval)
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


def updateLogZnXMomentsFinal(nFinal, logEofZ, logEofZ2, logEofX, LLhood_im1,
                             LLhood_i, trapezoidalFlag, errorEval):
    """
    Wrapper around updateZnXMomentsF taking into account whether trapezium rule is used or not
    """
    if trapezoidalFlag:
        logEofZ, logEofZ2, logEofWeight = updateLogZnXMomentsF(
            nFinal, logEofZ, logEofZ2, logEofX,
            np.log(0.5) + np.logaddexp(LLhood_im1, LLhood_i), errorEval)
    else:
        logEofZ, logEofZ2, logEofWeight = updateLogZnXMomentsF(
            nFinal, logEofZ, logEofZ2, logEofX, LLhood_i, errorEval)
    return logEofZ, logEofZ2, logEofWeight


def updateLogZnXMomentsF(nFinal, logEofZ, logEofZ2, logEofX, LL, errorEval):
    """
    TODO: rewrite docstring
    """
    if errorEval == 'recursive':
        logEofX = updateLogEofXFinal(logEofX, nFinal)
        logEofX2 = updateLogEofX2Final(logEofX, nFinal)
        logEofZ2 = updateLogEofZ2Final(logEofZ2, logEofX, logEofZ, logEofX2,
                                       LL)
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
    """
    try:
        return 1. / ZNew * weight * \
            np.log(Lhood) + Z / ZNew * (H + np.log(Z)) - np.log(ZNew)
    except FloatingPointError:  # take lim Z->0^+ Z / ZNew * (H + log(Z)) = 0
        return 1. / ZNew * weight * np.log(Lhood) - np.log(ZNew)


def updateHLog(H, logWeight, logZNew, LLhood, logZ):
    """
    update H using previous value, previous and new log(Z) and latest weight
    Isn't a non-log version as H propto log(L).
    As given in Skilling's paper
    TODO: consider if trapezium rule should lead to different implementation
    """
    try:
        return np.exp(logWeight - logZNew) * LLhood + \
            np.exp(logZ - logZNew) * (H + logZ) - logZNew
    # when logZ is -infinity, np.exp(logZ) * logZ cannot be evaluated. Treat
    # it as zero, ie treat it as lim Z->0^+ exp(logZ) * logZ = 0
    except FloatingPointError:
        return np.exp(logWeight - logZNew) * LLhood - logZNew


# calculate/ retrieve final estimates/ errors of Z


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
    return logsubexp(logEofX2, 2. * logEofX)


def calcVarZSkillingK(EofZ, nLive, H):
    """
    Uses definition of error given in Skilling's NS paper, ACCORDING to Keeton.
    Only valid in limit that Skilling's approximation of var[log(Z)] = H / nLive being correct,
    and E[Z]^2 >> var[Z] so that log(1+x)~x approximation is valid.
    Also requires that Z is log-normally distributed
    I think this is only valid for NS loop contributions, not final part or total
    """
    return EofZ**2. * H / nLive


def calcHSkillingK(EofZ, varZ, nLive):
    """
    Uses definition of error given in Skilling's NS paper, ACCORDING to Keeton.
    Only valid in limit that Skilling's approximation of var[log(Z)] = H / nLive being correct,
    and E[Z]^2 >> var[Z] so that log(1+x)~x approximation is valid.
    Also requires that Z is log-normally distributed
    I think this is only valid for NS loop contributions, not final part or total
    """
    return varZ * nLive / EofZ**2.


def calcVarLogZ(EofZ, varZ, method):
    """
    Uses propagation of uncertainty formula or
    relationship between log-normal r.v.s and the normally distributed log of the log-normal r.v.s
    to calculate var[logZ] from EofZ and varZ (taken from Wikipedia)
    """
    if method == 'uncertainty':
        return varZ / EofZ**2.
    elif method == 'log-normal':
        return np.sqrt(np.log(1. + varZ / EofZ**2.))


def calcEofLogZ(EofZ, varZ):
    """
    Calc E[log(Z)] from E[Z] and Var[Z]. Assumes Z is log-normally distributed
    """
    return np.log(EofZ**2. / (np.sqrt(varZ + EofZ**2.)))


def calcEofZ(EofLogZ, varLogZ):
    """
    calc E[Z] from E[logZ] and var[logZ]. Assumes Z is log-normal
    """
    return np.exp(EofLogZ + 0.5 * varLogZ)


def calcVarZ(varLogZ, method, EofZ=None, EofLogZ=None):
    """
    Uses propagation of uncertainty formula or
    relationship between log-normal r.v.s and the normally distributed log of the log-normal r.v.s
    to calculate var[Z] from EofZ and varLogZ (taken from Wikipedia)
    """
    if method == 'uncertainty':
        return varLogZ * EofZ**2.
    elif method == 'log-normal':
        return np.exp(2. * EofLogZ + varLogZ) * (np.exp(varLogZ) - 1.)


def calcVarLogZSkilling(H, nLive):
    """
    Skilling works in log space throughout, including calculating the moments of log(*)
    i.e. E[f(log(*))]. Thus he derives a value for the variance of log(Z), through his discussions of
    Poisson fluctuations whilst exploring the posterior.
    """
    return H / nLive


# Calculate Z moments and H a-posteri using Keeton's methods

# wrappers around E[t] functions for calculating powers of them.
# required for calculating Z moments with Keeton's method.


# E[t]^i
def EoftPowi(nLive, i):
    return calct(nLive)**i


# E[t^2]^i


def Eoft2Powi(nLive, i):
    return calct2(nLive)**i


# (E[t^2]/E[t])^i


def Eoft2OverEoftPowi(nLive, i):
    return (calct2(nLive) / calct(nLive))**i


def calcEofftArr(Eofft, nLive, n):
    """
    Calculates E[f(t)]^i then returns this with yield.
    Yield means next time function is called,
    it picks off from where it last returned,
    with same variable values as before returning.
    Note the function isn't executed until the generator return by yield is iterated over
    Putting for loop here is faster than filling in blank array
    """
    for i in range(1, n + 1):
        yield Eofft(nLive, i)


def getEofftArr(Eofft, nLive, nest):
    """
    faster than creating array of zeroes and looping over
    """
    return np.fromiter(calcEofftArr(Eofft, nLive, nest),
                       dtype=float,
                       count=nest)


def calcZMomentsKeeton(Lhoods, nLive, nest):
    """
    calculate Z moments a-posteri with full list of Lhoods used in NS loop,
    using equations given in Keeton
    """
    EofZ = calcEofZKeeton(Lhoods, nLive, nest)
    EofZ2 = calcEofZ2Keeton(Lhoods, nLive, nest)
    return EofZ, EofZ2


def calcEofZKeeton(Lhoods, nLive, nest):
    """
    Calculate first moment of Z from main NS loop.
    According to paper, this is just E[Z] = 1. / nLive * sum_i^nest L_i * E[t]^i
    """
    EoftArr = getEofftArr(EoftPowi, nLive, nest)
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
    EoftArr = getEofftArr(EoftPowi, nLive, nest)
    LEoft = Lhoods * EoftArr
    innerSums = np.fromiter(calcInnerSums(Lhoods, nLive, nest),
                            dtype=float,
                            count=nest)
    outerSums = LEoft * innerSums
    return outerSums.sum()


def calcInnerSums(Lhoods, nLive, nest):
    """
    Second generator (yielding) function, which returns inner sum for outer index k
    """
    for k in range(1, nest + 1):
        Eoft2OverEoftArr = getEofftArr(Eoft2OverEoftPowi, nLive, k)
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
            innerSum += Lhoods[i - 1] * Eoft2OverEoftPowi(nLive, i)
        outerSum = Lhoods[k - 1] * EoftPowi(nLive, k) * innerSum
        total += outerSum
    return total


def calcHKeeton(EofZ, Lhoods, nLive, nest):
    """
    Calculate H from KL divergence equation transformed to LX space
    as given in Keeton.
    """
    sumTerms = Lhoods * np.log(Lhoods) * getEofftArr(EoftPowi, nLive, nest)
    sumTerm = 1. / nLive * sumTerms.sum()
    return 1. / EofZ * sumTerm - np.log(EofZ)


def calcZMomentsKeetonLog(deadPointsLLhood, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZ, logEofZ2


def calcEofZKeetonLog(LLhoods, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZ


def calcEofZ2KeetonLog(LLhoods, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZ2


def calcHKeetonLog(logEofZK, deadPointsLLhood, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return H


def calcZMomentsFinalKeeton(finalLhoods, nLive, nest):
    """
    calculate Z moments a-posteri with list of final Lhood points (ones remaining at termination of main loop),
    using equations given in Keeton
    """
    EofZ = calcEofZFinalKeeton(finalLhoods, nLive, nest)
    EofZ2 = calcEofZ2FinalKeeton(finalLhoods, nLive, nest)
    return EofZ, EofZ2


def calcEofZFinalKeeton(finalLhoods, nLive, nest):
    """
    Averages over Lhood, which I don't think is the correct thing to do as it doesn't correspond to a unique parameter vector value.
    TODO: consider other ways of getting final contribution from livepoints with Keeton's method
    """
    LhoodAv = finalLhoods.mean()
    EofFinalX = EoftPowi(nLive, nest)
    return EofFinalX * LhoodAv


def calcEofZ2FinalKeeton(finalLhoods, nLive, nest):
    """
    Averages over Lhood, which I don't think is the correct thing to do as it doesn't correspond to a unique parameter vector value.
    TODO: consider other ways of getting final contribution from livepoints with Keeton's method
    """
    LhoodAv = finalLhoods.mean()
    EofFinalX2 = Eoft2Powi(nLive, nest)
    return LhoodAv**2. * EofFinalX2


def calcEofZZFinalKeeton(Lhoods, finalLhoods, nLive, nest):
    """
    Averages over Lhood for contribution from final points,
    which I don't think is the correct thing to do as it doesn't correspond to a unique parameter vector value.
    TODO: consider other ways of getting final contribution from livepoints with Keeton's method
    """
    finalLhoodAv = finalLhoods.mean()
    finalTerm = finalLhoodAv / (nLive + 1.) * EoftPowi(nLive, nest)
    Eoft2OverEoftArr = getEofftArr(Eoft2OverEoftPowi, nLive, nest)
    loopTerms = Lhoods * Eoft2OverEoftArr
    loopTerm = loopTerms.sum()
    return finalTerm * loopTerm


def calcHTotalKeeton(EofZ, Lhoods, nLive, nest, finalLhoods):
    """
    Calculates total value of H based on KL divergence equation transformed to
    LX space as given in Keeton.
    Uses H function used to calculate loop H value (but with total Z), and adapts
    final result to give HTotal
    """
    LAv = finalLhoods.mean()
    HPartial = calcHKeeton(EofZ, Lhoods, nLive, nest)
    return HPartial + 1. / EofZ * LAv * np.log(LAv) * EoftPowi(nLive, nest)


def calcZMomentsFinalKeetonLog(livePointsLLhood, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZFinal, logEofZ2Final


def calcEofZFinalKeetonLog(finalLLhoods, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZFinal


def calcEofZ2FinalKeetonLog(finalLLhoods, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZ2Final


def calcEofZZFinalKeetonLog(deadPointsLLhood, livePointsLLhood, nLive, nest):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZZFinalK


def calcHTotalKeetonLog(logEofZFinalK, deadPointsLLhood, nLive, nest,
                        livePointsLLhood):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return H


# Functions for combining contributions from main NS loop and termination
# ('final' quantities) for estimate or Z and its error


def getEofZTotalKeeton(EofZ, EofZFinal):
    """
    get total from NS loop and final contributions
    """
    return EofZ + EofZFinal


def getEofZ2TotalKeeton(EofZ2, EofZ2Final):
    """
    get total from NS loop and final contributions
    """
    return EofZ2 + EofZ2Final


def getVarTotalKeeton(varZ, varZFinal, EofZ, EofZFinal, EofZZFinal):
    """
    Get total variance from NS loop and final contributions.
    For recursive method, since E[ZLive] = E[ZTot] etc.,
    and assuming that the recurrence relations account for the covariance between
    Z and ZFinal, this is just varZFinal.
    For Keeton's method, have to explicitly account for correlation as expectations for Z and ZLive are essentially calculated independently
    TODO: check if recurrence relations of Z and ZFinal properly account for correlation between two
    """
    return varZ + varZFinal + 2. * (EofZZFinal - EofZ * EofZFinal)


def getVarTotalKeetonLog(logVarZ, logVarZFinal, logEofZ, logEofZFinal,
                         logEofZZFinal):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logVarZTotal


def getEofZTotalKeetonLog(logEofZ, logEofZFinal):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZTotal


def getEofZ2TotalKeetonLog(logEofZ2, logEofZ2Final):
    """
    TODO
    """
    print("not implemented yet. Exiting")
    sys.exit(1)
    return logEofZ2Total


# DEPRECATED I THINK


def getLogEofXLogEofw(nLive, X):
    """
    get increment (part of weight for posterior and evidence calculations) based on previous value of X, calculates latest X using t calculated from either expected value or sampling. Expected value can be of t (E[t]) or log(t) E[log(t)]. These are roughly the same for large nLive
    Sampling can take two forms: sampling from the pdf or taking the highest of U[0,1]^Nlive values (from which the pdf form is derived from), so they should in theory be the same.
    """
    expectation = 't'
    t = calct(nLive, expectation)
    XNew = X * t
    return np.log(XNew), np.log(X - XNew)


# DEPRECATED I THINK


def getLogEofWeight(logw, LLhood_im1, LLhood_i, trapezoidalFlag):
    """
    calculates logw + log(f(L_im1, L_i)) where f(L_im1, L_i) = L_i for standard quadrature
    and f(L_im1, L_i) = (L_im1 + L_i) / 2. for the trapezium rule
    """
    if trapezoidalFlag:
        # from Will's implementation, Z = sum (X_im1 - X_i) * 0.5 * (L_i +
        # L_im1)
        return np.log(0.5) + logw + np.logaddexp(LLhood_im1, LLhood_i)
    else:
        # weight of deadpoint (for posterior) = prior mass decrement *
        # likelihood
        return logw + LLhood_i


# plotting functions


def plotPhysPosteriorIW(x, unnormalisedSamples, Z, space):
    """
    Plots posterior in physical space according to importance weights w(theta)L(theta) / Z. Doesn't use KDE so isn't true shape of posterior.
    If inputting logWeights/ logZ then set space == 'log'
    """
    if space == 'log':
        normalisedSamples = np.exp(unnormalisedSamples - Z)
    else:
        normalisedSamples = unnormalisedSamples / Z
    plt.figure('phys posterior')
    plt.scatter(x, normalisedSamples)
    plt.show()
    plt.close()


def plotXPosterior(X, L, Z, space):
    """
    Plots X*L(X)/Z in log X space, not including KDE methods
    """
    if space == 'log':
        LhoodDivZ = np.exp(L - Z)
        X = np.exp(X)
    else:
        LhoodDivZ = L / Z
    LXovrZ = X * LhoodDivZ
    plt.figure('posterior')
    plt.scatter(X, LXovrZ)
    plt.set_xscale('log')
    plt.show()
    plt.close()


def callGetDist(chainsFilePrefix, plotName, nParams):
    """
    produces triangular posterior plots using getDist for first nParams
    parameters from chains file as labelled in that file and in .paramnames
    """
    paramList = ['p' + str(i + 1) for i in range(nParams)]
    chains = getdist.loadMCSamples(chainsFilePrefix)
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot([chains], paramList, filled_compare=True)
    g.export(plotName)


# print output functions


def printUpdate(nest, deadPointPhys, deadPointLhood, EofZ, livePointPhys,
                livePointLhood, space):
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
        print("invalid space")
        sys.exit(1)
    print("for deadpoint %i: physical value = %s %s value = %f" %
          (nest, deadPointPhys, L, deadPointLhood))
    print("%s = %s" % (Z, EofZ))
    print("new live point obtained: physical value = %s %s has value = %s" %
          (livePointPhys, L, livePointLhood))


def printBreak():
    """
    tell user final contribution to sampling is being calculated
    """
    print("adding final contribution from remaining live points")


def printZHValues(EofZ, EofZ2, varZ, H, space, stage, method):
    """
    print values of Z (including varios moments, variance) and H
    in either log or linear space, at a given stage and calculated by a given method
    """
    if space == 'log':
        Z = 'ln(E[Z])'
        Z2 = 'ln([Z^2])'
        var = 'ln(var[Z])'
    elif space == 'linear':
        Z = 'E[Z]'
        Z2 = 'E[Z2]'
        var = 'var[Z]'
    else:
        print("invalid space")
        sys.exit(1)
    print("%s %s (%s) = %s" % (Z, stage, method, EofZ))
    print("%s %s (%s) = %s" % (Z2, stage, method, EofZ2))
    print("%s %s (%s) = %s" % (var, stage, method, varZ))
    print("H %s (%s) = %s" % (stage, method, H))


def printTheoretical(ZTheor, ZTheorErr, HTheor, HTheorErr):
    """
    Outputs values for theoretical values of Z and H (and their errors)
    """
    print("ZTheor = %s" % ZTheor)
    print("ZTheorErr = %s" % ZTheorErr)
    print("HTheor = %s" % HTheor)
    print("HTheorErr = %s" % HTheorErr)


def printSampleNum(numSamples):
    """
    Print number of samples used in sampling (including final livepoints used for posterior weights)
    """
    print("total number of samples = %i" % numSamples)


def printTerminationUpdateInfo(nest, terminator):
    """
    Print update on termination status when evaluating by H value
    """
    print("current end value is %i. Termination value is %f" %
          (nest, terminator))


def printTerminationUpdateZ(EofZLive, endValue, terminationFactor, space):
    """
    Print update on termination status when evaluating by Z ratio
    """
    if space == 'linear':
        Z = 'E[ZLive]'
    elif space == 'log':
        Z = 'log(E[ZLive])'
    else:
        print("invalid space")
        sys.exit(1)
    print("%s = %s" % (Z, EofZLive))
    print("current end value is %s. Termination value is %s" %
          (endValue, terminationFactor))


def printFinalLivePoints(i, physValue, Lhood, ZLiveType, space):
    """
    print information about final livepoints used to calculate final
    contribution to Z/ posterior samples.
    """
    if space == 'linear':
        L = 'Lhood'
    elif space == 'log':
        if ZLiveType == 'average Lhood':
            L = 'log(average Lhood)'
        else:
            L = 'LLhood'
    else:
        print("invalid space")
        sys.exit(1)
    if ZLiveType == 'average Lhood':
        print(
            "'average' physical value = %s (n.b. this has no useful meaning), %s = %s"
            % (physValue, L, Lhood))
    elif ZLiveType == 'average X':
        print(
            "remaining livepoint number %i: physical value = %s %s value = %s"
            % (i, physValue, L, Lhood))
    elif ZLiveType == 'max Lhood':
        print(
            "maximum %s remaining livepoint number %i: physical value = %s %s value = %s"
            % (L, i, physValue, L, Lhood))


# nested run functions


def NestedRun(priorParams, LLhoodParams, paramNames, setupDict):
    """
    function which completes a NS run. parameters of priors and likelihood need to be specified, as well as a flag indication type of prior for each dimension and the pdf for the lhood.
    setupDict contains other setup parameters such as termination type & factor, method of finding new livepoint, details of how weights are calculated, how final Z contribution is added, and directory/file prefix for saved files.
    """
    nLive = 50
    nDims = len(paramNames)
    checkInputParamsShape(priorParams, LLhoodParams, nDims)
    # initialise livepoints to random values uniformly on [0,1]^D
    livePoints = np.random.rand(nLive, nDims)
    priorObjs = fitPriors(priorParams)
    priorFuncsPdf = getPriorPdfs(priorObjs)
    priorFuncsPpf = getPriorPpfs(priorObjs)
    # Convert livepoint values to physical values
    livePointsPhys = invPrior(livePoints, priorFuncsPpf)
    LhoodObj = fitLhood(LLhoodParams)
    LLhoodFunc = LLhood(LhoodObj)
    # calculate LLhood values of initial livepoints
    livePointsLLhood = LLhoodFunc(livePointsPhys)
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
    # begin nested sample loop
    while True:
        LLhoodStarOld = LLhoodStar
        # index of lowest likelihood livepoint (next deadpoint)
        deadIndex = np.argmin(livePointsLLhood)
        # LLhood of dead point and new target
        LLhoodStar = livePointsLLhood[deadIndex]
        # update expected values of moments of X and Z, and get posterior
        # weights
        logEofZNew, logEofZ2, logEofZX, logEofX, logEofX2, logEofWeight = updateLogZnXMoments(
            nLive, logEofZ, logEofZ2, logEofZX, logEofX, logEofX2,
            LLhoodStarOld, LLhoodStar, setupDict['trapezoidalFlag'])
        logEofXArr.append(logEofX)
        logEofWeights.append(logEofWeight)
        H = updateHLog(H, logEofWeight, logEofZNew, LLhoodStar, logEofZ)
        logEofZ = logEofZNew  # update evidence part II
        # WARNING, VIEWING A NUMPY SLICE (IE NOT USING NP.COPY) DOES NOT CREATE A COPY AND SO A-POSTORI CHANGES TO ARRAY WILL AFFECT PREVIOUSLY SLICED ARRAY
        # USE NP.MAY_SHARE_MEMORY(A, B) TO SEE IF ARRAYS SHARE MEMORY, PYTHON
        # 'IS' KEYWORD DOESN'T WORK
        deadPointPhys = np.copy(livePointsPhys[deadIndex]).reshape(1, -1)
        deadPointsPhys.append(deadPointPhys)
        deadPointLLhood = LLhoodStar
        deadPointsLLhood.append(deadPointLLhood)
        # update array where last deadpoint was with new livepoint picked
        # subject to L_new > L*
        if setupDict['sampler'] == 'blind':
            livePointsPhys[deadIndex], livePointsLLhood[
                deadIndex] = getNewLiveBlind(priorFuncsPpf, LLhoodFunc,
                                             LLhoodStar)
        elif setupDict['sampler'] == 'MH':
            livePointsPhys[deadIndex], livePointsLLhood[
                deadIndex] = getNewLiveMH(livePointsPhys, deadIndex,
                                          priorFuncsPdf, priorParams,
                                          LLhoodFunc, LLhoodStar)
        if setupDict['verbose']:
            printUpdate(nest, deadPointPhys, deadPointLLhood, logEofZ,
                        livePointsPhys[deadIndex].reshape(1, -1),
                        livePointsLLhood[deadIndex], 'log')
        nest += 1
        if nest % checkTermination == 0:
            breakFlag, liveMaxIndex, liveLLhoodMax, avLLhood, nFinal = tryTerminationLog(
                setupDict['verbose'], setupDict['terminationType'],
                setupDict['terminationFactor'], nest, nLive, logEofX,
                livePointsLLhood, LLhoodStar, setupDict['ZLiveType'],
                setupDict['trapezoidalFlag'], logEofZ, H)
            if breakFlag:  # termination condition was reached
                break
    EofZ = np.exp(logEofZ)
    EofZ2 = np.exp(logEofZ2)
    varZ = calcVariance(EofZ, EofZ2)
    EofZK, EofZ2K = calcZMomentsKeeton(np.exp(np.array(deadPointsLLhood)),
                                       nLive, nest)
    varZK = calcVariance(EofZK, EofZ2K)
    HK = calcHKeeton(EofZK, np.exp(np.array(deadPointsLLhood)), nLive, nest)
    if setupDict['verbose']:
        printBreak()
        printZHValues(EofZ, EofZ2, varZ, H, 'linear', 'before final',
                      'recursive')
        printZHValues(EofZK, EofZ2K, varZK, HK, 'linear', 'before final',
                      'Keeton equations')
    logEofZTotal, logEofZ2Total, H, livePointsPhysFinal, livePointsLLhoodFinal, logEofXFinalArr = getFinalContributionLog(
        setupDict['verbose'], setupDict['ZLiveType'],
        setupDict['trapezoidalFlag'], nFinal, logEofZ, logEofZ2, logEofX,
        logEofWeights, H, livePointsPhys, livePointsLLhood, avLLhood,
        liveLLhoodMax, liveMaxIndex, LLhoodStar)
    totalPointsPhys, totalPointsLLhood, logEofXArr, logEofWeights = getTotal(
        deadPointsPhys, livePointsPhysFinal, deadPointsLLhood,
        livePointsLLhoodFinal, logEofXArr, logEofXFinalArr, logEofWeights)
    EofZTotal = np.exp(logEofZTotal)
    EofZ2Total = np.exp(logEofZ2Total)
    varZ = calcVariance(EofZTotal, EofZ2Total)
    EofZFinalK, EofZ2FinalK = calcZMomentsFinalKeeton(np.exp(livePointsLLhood),
                                                      nLive, nest)
    varZFinalK = calcVariance(EofZFinalK, EofZ2FinalK)
    EofZZFinalK = calcEofZZFinalKeeton(np.exp(np.array(deadPointsLLhood)),
                                       np.exp(livePointsLLhood), nLive, nest)
    varZTotalK = getVarTotalKeeton(varZK, varZFinalK, EofZK, EofZFinalK,
                                   EofZZFinalK)
    EofZTotalK = getEofZTotalKeeton(EofZK, EofZFinalK)
    EofZ2TotalK = getEofZ2TotalKeeton(EofZ2K, EofZ2FinalK)
    HK = calcHTotalKeeton(EofZTotalK, np.exp(np.array(deadPointsLLhood)),
                          nLive, nest, np.exp(livePointsLLhood))
    priorFuncsLogPdf = getPriorLogPdfs(priorObjs)
    ZTheor, ZTheorErr, priorVolume = calcZTheor(priorParams, priorFuncsLogPdf,
                                                LLhoodFunc, nDims)
    HTheor, HTheorErr = calcHTheor(priorParams, priorFuncsPdf, LLhoodFunc,
                                   nDims, ZTheor, ZTheorErr)
    numSamples = len(totalPointsPhys[:, 0])
    if setupDict['verbose']:
        printZHValues(EofZTotal, EofZ2Total, varZ, H, 'linear', 'total',
                      'recursive')
        printZHValues(EofZFinalK, EofZ2FinalK, varZFinalK, 'not calculated',
                      'linear', 'final contribution', 'Keeton equations')
        printZHValues(EofZTotalK, EofZ2TotalK, varZTotalK, HK, 'linear',
                      'total', 'Keeton equations')
        printTheoretical(ZTheor, ZTheorErr, HTheor, HTheorErr)
    if setupDict['outputFile']:
        writeOutput(setupDict['outputFile'], totalPointsPhys,
                    totalPointsLLhood, logEofWeights, logEofXArr, paramNames,
                    'log')
    return logEofZ, totalPointsPhys, totalPointsLLhood, logEofWeights, logEofXArr


def NestedRunLinear(priorParams, LhoodParams, paramNames, setupDict):
    """
    function which completes a NS run. parameters of priors and likelihood need to be specified, as well as a flag indication type of prior for each dimension and the pdf for the lhood.
    setupDict contains other setup parameters such as termination type & factor, method of finding new livepoint, details of how weights are calculated, how final Z contribution is added, and directory/file prefix for saved files.
    """
    nLive = 50
    nDims = len(paramNames)
    checkInputParamsShape(priorParams, LhoodParams, nDims)
    # initialise livepoints to random values uniformly on [0,1]^D
    livePoints = np.random.rand(nLive, nDims)
    priorObjs = fitPriors(priorParams)
    priorFuncsPdf = getPriorPdfs(priorObjs)
    priorFuncsPpf = getPriorPpfs(priorObjs)
    # Convert livepoint values to physical values
    livePointsPhys = invPrior(livePoints, priorFuncsPpf)
    LhoodObj = fitLhood(LhoodParams)
    LhoodFunc = Lhood(LhoodObj)
    # calculate LLhood values of initial livepoints
    livePointsLhood = LhoodFunc(livePointsPhys)
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
    # begin nested sample loop
    while True:
        LhoodStarOld = LhoodStar
        # index of lowest likelihood livepoint (next deadpoint)
        deadIndex = np.argmin(livePointsLhood)
        # LLhood of dead point and new target
        LhoodStar = livePointsLhood[deadIndex]
        # update expected values of moments of X and Z, and get posterior
        # weights
        EofZNew, EofZ2, EofZX, EofX, EofX2, EofWeight = updateZnXMoments(
            nLive, EofZ, EofZ2, EofZX, EofX, EofX2, LhoodStarOld, LhoodStar,
            setupDict['trapezoidalFlag'])
        EofXArr.append(EofX)
        EofWeights.append(EofWeight)
        H = updateH(H, EofWeight, EofZNew, LhoodStar, EofZ)
        EofZ = EofZNew  # update evidence part II
        # WARNING, VIEWING A NUMPY SLICE (IE NOT USING NP.COPY) DOES NOT CREATE A COPY AND SO A-POSTORI CHANGES TO ARRAY WILL AFFECT PREVIOUSLY SLICED ARRAY
        # USE NP.MAY_SHARE_MEMORY(A, B) TO SEE IF ARRAYS SHARE MEMORY, PYTHON
        # 'IS' KEYWORD DOESN'T WORK
        deadPointPhys = np.copy(livePointsPhys[deadIndex]).reshape(1, -1)
        deadPointsPhys.append(deadPointPhys)
        deadPointLhood = LhoodStar
        deadPointsLhood.append(deadPointLhood)
        # update array where last deadpoint was with new livepoint picked
        # subject to L_new > L*
        if setupDict['sampler'] == 'blind':
            livePointsPhys[deadIndex], livePointsLhood[
                deadIndex] = getNewLiveBlind(priorFuncsPpf, LhoodFunc,
                                             LhoodStar)
        elif setupDict['sampler'] == 'MH':
            livePointsPhys[deadIndex], livePointsLhood[
                deadIndex] = getNewLiveMH(livePointsPhys, deadIndex,
                                          priorFuncsPdf, priorParams,
                                          LhoodFunc, LhoodStar)
        if setupDict['verbose']:
            printUpdate(nest, deadPointPhys, deadPointLhood, EofZ,
                        livePointsPhys[deadIndex].reshape(1, -1),
                        livePointsLhood[deadIndex], 'linear')
        nest += 1
        if nest % checkTermination == 0:
            breakFlag, liveMaxIndex, liveLhoodMax, avLhood, nFinal = tryTermination(
                setupDict['verbose'], setupDict['terminationType'],
                setupDict['terminationFactor'], nest, nLive, EofX,
                livePointsLhood, LhoodStar, setupDict['ZLiveType'],
                setupDict['trapezoidalFlag'], EofZ, H)
            if breakFlag:  # termination condition was reached
                break
    varZ = calcVariance(EofZ, EofZ2)
    EofZK, EofZ2K = calcZMomentsKeeton(np.array(deadPointsLhood), nLive, nest)
    varZK = calcVariance(EofZK, EofZ2K)
    HK = calcHKeeton(EofZK, np.array(deadPointsLhood), nLive, nest)
    if setupDict['verbose']:
        printBreak()
        printZHValues(EofZ, EofZ2, varZ, H, 'linear', 'before final',
                      'recursive')
        printZHValues(EofZK, EofZ2K, varZK, HK, 'linear', 'before final',
                      'Keeton equations')
    EofZTotal, EofZ2Total, H, livePointsPhysFinal, livePointsLhoodFinal, EofXFinalArr = getFinalContribution(
        setupDict['verbose'], setupDict['ZLiveType'],
        setupDict['trapezoidalFlag'], nFinal, EofZ, EofZ2, EofX, EofWeights, H,
        livePointsPhys, livePointsLhood, avLhood, liveLhoodMax, liveMaxIndex,
        LhoodStar)
    totalPointsPhys, totalPointsLhood, EofXArr, EofWeights = getTotal(
        deadPointsPhys, livePointsPhysFinal, deadPointsLhood,
        livePointsLhoodFinal, EofXArr, EofXFinalArr, EofWeights)
    varZ = calcVariance(EofZTotal, EofZ2Total)
    EofZFinalK, EofZ2FinalK = calcZMomentsFinalKeeton(livePointsLhood, nLive,
                                                      nest)
    varZFinalK = calcVariance(EofZFinalK, EofZ2FinalK)
    EofZZFinalK = calcEofZZFinalKeeton(np.array(deadPointsLhood),
                                       livePointsLhood, nLive, nest)
    varZTotalK = getVarTotalKeeton(varZK, varZFinalK, EofZK, EofZFinalK,
                                   EofZZFinalK)
    EofZTotalK = getEofZTotalKeeton(EofZK, EofZFinalK)
    EofZ2TotalK = getEofZ2TotalKeeton(EofZ2K, EofZ2FinalK)
    HK = calcHTotalKeeton(EofZTotalK, np.array(deadPointsLhood), nLive, nest,
                          livePointsLhood)
    LLhoodFunc = LLhood(LhoodObj)
    priorFuncsLogPdf = getPriorLogPdfs(priorObjs)
    ZTheor, ZTheorErr, priorVolume = calcZTheor(priorParams, priorFuncsLogPdf,
                                                LLhoodFunc, nDims)
    HTheor, HTheorErr = calcHTheor(priorParams, priorFuncsPdf, LLhoodFunc,
                                   nDims, ZTheor, ZTheorErr)
    numSamples = len(totalPointsPhys[:, 0])
    if setupDict['verbose']:
        printZHValues(EofZTotal, EofZ2Total, varZ, H, 'linear', 'total',
                      'recursive')
        printZHValues(EofZFinalK, EofZ2FinalK, varZFinalK, 'not calculated',
                      'linear', 'final contribution', 'Keeton equations')
        printZHValues(EofZTotalK, EofZ2TotalK, varZTotalK, HK, 'linear',
                      'total', 'Keeton equations')
        printTheoretical(ZTheor, ZTheorErr, HTheor, HTheorErr)
    if setupDict['outputFile']:
        writeOutput(setupDict['outputFile'], totalPointsPhys, totalPointsLhood,
                    EofWeights, EofXArr, paramNames, 'linear')
    return EofZ, totalPointsPhys, totalPointsLhood, EofWeights, EofXArr


# main function


def main():
    # samplingFlag and maxPoints flag are explained in getIncrement(...)
    # function
    setupDict = {
        'verbose': True,
        'trapezoidalFlag': False,
        'ZLiveType': 'average Lhood',
        'terminationType': 'evidence',
        'terminationFactor': 0.5,
        'sampler': 'MH',
        'outputFile': './output/test'
    }
    # priorParams is (3,nDims) shape array. For a given parameter, first value indicates prior type (1 =	UNIFORM, 2 = NORMAL)
    # for UNIFORM PDF, 2nd value is lower bound, 3rd value is upper bound
    # for NORMAL PDF, 2nd value is mean, 3rd value is variance (parameter priors are assumed to be INDEPENDENT)
    # priorParams = np.array([[1, -5., 5.], [1, -5., 5.], [1, -5., 5.], [1, -5., 5.]]).T
    priorParams = np.array([[1, -5., 5.], [1, -5., 5.]]).T
    # LLhoodParams has shape (1, (1, nDims), (nDims, nDims)). First value (scalar) indicates type of likelihood function (2 = NORMAL)
    # second element (shape (1, nDims)) is the mean value for the likelihood in each dimension.
    # third element (shape (nDims, nDims)) is the covariance matrix for the
    # likelihood
    LLhoodParams = [
        2,
        np.array([0., 0.]).reshape(1, 2),
        np.array([1., 0., 0., 1.]).reshape(2, 2)
    ]
    # LLhoodParams = [2, np.array([0., 0., 0., 0.]).reshape(1,4), np.array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]).reshape(4,4)]
    # paramNames = ['\\theta_1', '\\theta_2', '\\theta_3', '\\theta_4']
    paramNames = ['\\theta_1', '\\theta_2']
    # ensures numpy raises FloatingPointError associated with exp(-inf)*-inf
    np.seterr(all='raise')
    # set this to value if you want NS to use same randomisations
    np.random.seed(0)
    # logEofZ, totalPointsPhys, totalPointsLLhood, logWeights, Xarr = NestedRun(priorParams, LLhoodParams, paramNames, setupDict)
    EofZ, totalPointsPhys, totalPointsLhood, weights, Xarr = NestedRun(
        priorParams, LLhoodParams, paramNames, setupDict)
    # callGetDist('./output/MH_gauss_uniform_4D', './plots/MH_gauss_uniform_4D', len(paramNames))
    # plotXPosterior(Xarr, totalPointsLLhood, logZ)


if __name__ == '__main__':
    main()
