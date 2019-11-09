# import standard modules
import numpy as np
import scipy
try:  # newer scipy versions
    from scipy.special import logsumexp
except ImportError:  # older scipy versions
    from scipy.misc import logsumexp

# import custom modules
from . import output
from . import tools

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
            output.printTerminationUpdateInfo(nest, terminator)
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
            output.printTerminationUpdateZ(logEofZLive, endValue,
                                           terminationFactor, 'log')
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
            output.printTerminationUpdateInfo(nest, terminator)
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
            output.printTerminationUpdateZ(EofZLive, endValue,
                                           terminationFactor, 'linear')
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
    # logEofZLive = tools.logAddArr2(-np.inf, logEofWeightsLive)
    logEofZLive = logsumexp(logEofWeightsLive)
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
            # LSumLhood = np.array([tools.logAddArr2(-np.inf, liveLLhoods)])
            # Make array for consistency
            LSumLhood = np.array([logsumexp(liveLLhoods)])
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
            # LSumLhood = np.array([tools.logAddArr2(-np.inf, liveLLhoods)])
            LSumLhood = np.array([logsumexp(liveLLhoods)])
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
