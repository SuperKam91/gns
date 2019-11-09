# import standard modules
import numpy as np
import scipy.integrate
import scipy
try:  # newer scipy versions
    from scipy.special import logsumexp
except ImportError:  # older scipy versions
    from scipy.misc import logsumexp

# import custom modules

# Z & H theoretical functions


def nDIntegratorZTheor(integrandFuncs,
                       limitsList,
                       integrandLogVal,
                       divByFact=True):
    """
    integrator used for calculating theoretical value of Z. integrand is function which evaluates to value of integrand at given parameter values (which are determined by nquad function).
    integrandFuncs is list of functions (Lhood & non-rectangular priors) which are multiplied together to give value of integrand.
    If integrandLogVal evaluates to true, does integration method which takes exp of log of integrand * some large number given by integrandLogVal
    to avoid underflow. The final integral result (and error) is then divided by exp(integrandLogVal) to get the final value.
    If divByFact == True, divides integral by exp(integrandLogVal) to give integral of function. If not leaves this step out, and essentially gives integral of function * exp(integrandLogVal). The latter can be useful when exp(integrandLogVal) would under/overflow, in which case one should consider log(integral of function) which is given by log(integral of function * exp(integrandLogVal)) - integrandLogVal

    Args:

    integrandFuncs : list functions which form integrand (through their product)

    limitsList : list integration limits

    integrandLogVal : boolean whether to take log of integrand or not

    divByFact : boolean whether to divide by log of integrandLogVal or not

    """
    if integrandLogVal:
        if divByFact:
            divByFact = np.exp(integrandLogVal)
        else:
            divByFact = 1.
        return scipy.integrate.nquad(
            evalExpLogIntegrand,
            limitsList,
            args=(integrandFuncs, integrandLogVal)) / divByFact
    else:  # this should only occur if you aren't concerned about underflow
        return scipy.integrate.nquad(evalIntegrand,
                                     limitsList,
                                     args=(integrandFuncs, ))


def nDIntegratorHTheor(integrandFuncs,
                       limitsList,
                       integrandLogVal,
                       divByFact=True,
                       LLhoodFunc=None):
    """
    As above but has to call nquad with a  slightly different function representing integrand when using
    exp(log(integrand)) method for calculating, due to LLhood(theta) part of integrand

    Args:

    integrandFuncs : list functions which form integrand (through their product)

    limitsList : list integration limits

    integrandLogVal : boolean whether to take log of integrand or not

    divByFact : boolean whether to divide by log of integrandLogVal or not

    LLhoodFunc : function or False log likelihood function if to be included in integrand calculation

    """
    if integrandLogVal:
        if divByFact:
            divByFact = np.exp(integrandLogVal)
        else:
            divByFact = 1.
        return scipy.integrate.nquad(
            evalLogLExpLogIntegrand,
            limitsList,
            args=(integrandFuncs, integrandLogVal, LLhoodFunc)) / divByFact
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
    along with an arbitrary 'large' ('small') value given in paramsNIntLogFuncsNIntLogVal[-1]
    then exponentiates this value to avoid underflow (overflow) when evaluating the pdfs/ multiplying them together.
    This should avoid most underflow (overflow) at all, provided the given value of integrandLogVal is large (small) enough
    There is some underflow/ loss of precision you cannot avoid, regardless of the value of integrandLogVal.
    This is because the range of values the LLhood can take can be large e.g. O(-10^5) to ~ O(-10) (O(10^10) to ~ O(10^14)).
    In case of adding 'large' ('small') value to prevent underflow (overflow, be cautious as this could lead to large (small) function values overflowing (underflowing!
    Dynamically calculating integrandLogVal wouldn't help, as it needs to factor into integrand
    n.b. underflow of small function values is usually better than overflow of large function values, as the latter contributes towards the value of the integral the most.
    args consists of 1) the vector of theta values for given call (from nquad)
    2) list of functions which make up the integrand
    3) 'large' ('small') number to be added to logarithm before exponentiated to give integrand. This value should be based on codomain of function you are integrating.
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


def integrateLogFunc(logPriorFunc, LLhoodFunc, targetSupport):
    """
    Simple, inefficient n-dimensional integrator,
    which calculates integrand in log space (n.b. the spacing is still linear).
    Uses logaddexp to get log of integrand.
    Takes equally spaced samples in each dimension (so width in each dimension is upper - lower lim / num samples in dim).
    priorHyperParams is as explained in getTheoreticalSamples()

    Args:

    logPriorFunc : function log prior function

    LLhoodFunc : function or False log likelihood function if to be included in integrand calculation

    targetSupport : array target support values in array of shape (3, nDims)

    """
    sampleWidth = 1.
    oneDn = 100  # number of points per dimension
    nDims = len(targetSupport[0, :])
    n = oneDn**nDims
    oneDGrids = []
    for i in range(nDims):
        if np.isfinite(targetSupport[2, i]):
            lowerBound = targetSupport[0, i]
            upperBound = targetSupport[1, i]
        else:
            mu = priorHyperParams[0, i]
            sigma = priorHyperParams[1, i]
            lowerBound = mu - 10 * sigma
            upperBound = mu + 10 * sigma
        oneDGrids.append(np.linspace(lowerBound, upperBound, oneDn))
        sampleWidth *= (upperBound - lowerBound) / oneDn
    meshGrids = np.meshgrid(*oneDGrids)
    params = np.hstack((meshGrid.reshape(-1, 1) for meshGrid in meshGrids))
    logPrior = np.zeros(n)
    for i in range(
            n
    ):  # there should be a better way of doing this. But as it stands, logPriorFunc only works on (nDim,) or (1, nDim) arrays
        logPrior[i] = logPriorFunc(params[i, :])
    LLhood = LLhoodFunc(params).reshape(-1, )
    logIntegrandArr = logPrior + LLhood
    logIntegral = logsumexp(logIntegrandArr) + \
        np.log(sampleWidth)  # change to scipy.special
    return logIntegral


def getPriorIntegrandAndLimits(priorFunc,
                               targetSupport,
                               integrandFuncs,
                               integrateAll=True,
                               priorFuncsPdf=None):
    """
    Adds prior functions to integrandFuncs dict for 'non-rectangular' (uniform) dimensions, as long as a mapping to dimensions of data required to integrate over for that function. For uniform priors, calculates the hyperrectangular volume. Also creates list of limits in each dimension of integral. For Gaussian priors, sets limits to +- infinity
    Integrate all means integrate across all dimensions, even if it has a uniform prior.
    Also uses priorFunc (i.e evaluates all priors at once) rather than priorFuncsPdf. Means that it can be used with user priors and Lhoods
    Not much slower than integrateAll = False, as that method still has to integrate Lhood over all dimensions (just not prior)
    integrateAll = False assumes all finitely bounded prior dimensions are rectangular, and works out their prior volume
    as being a rectangle. Also uses priorFuncsPdf, so can't be used with user defined functions. NOTE THIS WILL NOT GIVE THE CORRECT RESULT IF ANY OF THE TOY PRIORS ARE BOUNDED BUT AREN'T UNIFORM, AS WHEN CHECKING TARGETSUPPORT IT WILL ASSUME THAT DIMENSION IS UNIFORM. Could be fixed by adding a flag to targetSupport as to whether prior is uniform or not, but this method is sort of deprecated anyway.

    Args:

    priorFunc : function prior function

    targetSupport : array target support values in array of shape (3, nDims)

    integrandFuncs : list functions which form integrand (through their product)

    integrateAll : boolean see body of docstring

    priorFuncsPdf : list or False whether to include prior functions in integrand or not

    """
    hyperRectangleVolume = 1.
    limitsList = []
    if integrateAll:
        for i in range(len(targetSupport[0, :])):
            limitsList.append(
                np.array([targetSupport[0, i], targetSupport[1, i]]))
        integrandFuncs[priorFunc] = slice(None)
    else:
        for i in range(len(targetSupport[0, :])):
            limitsList.append(
                np.array([targetSupport[0, i], targetSupport[1, i]]))
            if np.isfinite(targetSupport[2, i]):
                priorWidth = targetSupport[2, i]
                hyperRectangleVolume *= priorWidth
            else:
                # tuple representing slice of data array x for this prior
                # function, is mapped to that prior function via the dictionary
                integrandFuncs[priorFuncsPdf[i]] = (0, i)
    return integrandFuncs, limitsList, hyperRectangleVolume


class ZTheorException(Exception):
    """
    To be used if Ztheor can't be calculated due to for example, high dimensionality
    """
    pass


def calcZTheor(priorFunc,
               LLhoodFunc,
               targetSupport,
               nDims,
               integrandLogVal=1.,
               LhoodFunc=None):
    """
    numerically integrates L(theta) * pi(theta) over theta
    priorFuncs must be in same order as dimensions of LhoodFunc when it was fitted (and in same order as priorParams).
    A bit slow, but not sure how I can make it faster tbh, as it will get exponentially slower with # of dimensions
    LhoodFunc and priorFuncsPdf can be .pdf() or .logpdf() methods (but must be same), but must alter value of integrandLogVal accordingly (set to finite number if want to to integration in log space, something which evaluates to False otherwise)

    Args:

    priorFunc : function prior function

    targetSupport : array target support values in array of shape (3, nDims)

    LLhoodFunc : function log likelihood function if to be included in integrand calculation (if log likelihood not provided)

    nDims : int number of dimensions of integral

    integrandLogVal : boolean whether to take log of integrand or not

    LhoodFunc : function or False likelihood function if to be included in integrand calculation (instead of likelihood)

    """
    if LhoodFunc:  # calculate Z without considering underflow
        # slice refers to which dimensions of data array are required for given
        # function in integration call. In case of Lhood, all dimensions of the
        # parameter space are required
        integrandFuncs = {LhoodFunc: slice(None)}
    else:
        # evaluate L(theta) using exp(log(L(theta)))
        integrandFuncs = {LLhoodFunc: slice(None)}
    integrandFuncs, limitsList, hyperRectangleVolume = getPriorIntegrandAndLimits(
        priorFunc, targetSupport, integrandFuncs)
    ZIntegral, ZIntegralE = nDIntegratorZTheor(integrandFuncs, limitsList,
                                               integrandLogVal)
    ZTheor = 1. / hyperRectangleVolume * ZIntegral
    return ZTheor, ZIntegralE


def calcZTheorApprox(targetSupport):
    """
    Only valid in limit that prior is hyperrectangle, and majority of lhood is contained in prior hypervolume
    such that limits of integration (domain of the sampling space defined by the prior) can be extended close enough +- infinity such that the Lhood integrates to 1 over this domain

    Args:

    targetSupport : array target support values in array of shape (3, nDims)

    """
    priorVolume = 1.
    for i in range(len(targetSupport[0, :])):
        if np.isfinite(targetSupport[2, i]):
            priorVolume *= targetSupport[2, i]
    ZTheor = 1. / priorVolume
    return ZTheor, priorVolume


def calcHTheor(priorFunc,
               LLhoodFunc,
               targetSupport,
               nDims,
               Z,
               ZErr,
               integrandLogVal=None,
               LhoodFunc=None):
    """
    Calculates HTheor from the KL divergence equation: H = int[P(theta) * ln(P(theta) / pi(theta))] = 1/Z * int[L(theta) * pi(theta) * ln(L(theta))] - ln(Z).
    For uniform priors, calculates volume and skips that part of integral (over pi(theta)).
    Uses same trick as ZTheor in that it composes a dictionary of functions to integrate, mapped to dimension(s) of theta vector to integrate along for given function.
    Passes this dictionary to function which nquad actually evaluates.

    Args:

    priorFunc : function prior function

    LLhoodFunc : function log likelihood function if to be included in integrand calculation (if log likelihood not provided)

    targetSupport : array target support values in array of shape (3, nDims)

    nDims : int number of dimensions of integral

    Z : float Bayesian evidence

    ZErr : float error on Z

    integrandLogVal : boolean whether to take log of integrand or not

    LhoodFunc : function or False likelihood function if to be included in integrand calculation (instead of likelihood)

    """
    if LhoodFunc:  # calculate H without considering underflow
        # slice refers to which dimensions of data array are required for given
        # function in integration call. In case of Lhood, all dimensions of the
        # parameter space are required
        integrandFuncs = {LhoodFunc: slice(None), LLhoodFunc: slice(None)}
    else:
        # evaluate L(theta) using exp(log(L(theta)))
        integrandFuncs = {LLhoodFunc: slice(None)}
    integrandFuncs, limitsList, hyperRectangleVolume = getPriorIntegrandAndLimits(
        priorFunc, targetSupport, integrandFuncs)
    LhoodPiLogLIntegral, LhoodPiLogLErr = nDIntegratorHTheor(
        integrandFuncs, limitsList, integrandLogVal, LLhoodFunc)
    HErr = calcHErr(Z, ZErr, LhoodPiLogLIntegral, LhoodPiLogLErr)
    return 1. / (hyperRectangleVolume * Z) * \
        LhoodPiLogLIntegral - np.log(Z), HErr


def calcHErr(Z, ZErr, LhoodPiLogLIntegral, LhoodPiLogLErr):
    """
    Calculates error on H due to uncertainty of Z, HIntegrand and ln(Z).
    ignores possible correlation between Z and LhoodPiLogLIntegral

    Args:

    Z : float Bayesian evidence

    ZErr : float error on Z

    LhoodPiLogLIntegral : float log of integral of likelihood x prior

    LhoodPiLogLErr : float error on log of integral of likelihood x prior

    """
    logZErr = ZErr / Z
    IntOverZErr = LhoodPiLogLIntegral / Z * \
        np.sqrt((ZErr / Z)**2. + (LhoodPiLogLErr / LhoodPiLogLIntegral)**2.)
    return np.sqrt(logZErr**2. + IntOverZErr**2.)


def calcHTheorApprox(Z, nDims, priorVolume):
    """
    Only valid in limit that prior is hyperrectangle, and majority of lhood is contained in prior hypervolume
    such that limits of integration can be extended to +- infinity
    ONLY VALID FOR NON-TRUNCATED GAUSSIAN LIKELIHOODS
    TODO: consider approximations for non-Gaussian Lhoods

    Args:

    Z : float Bayesian evidence

    nDims : int number of dimensions of integral

    priorVolume : float volume of hyper rectangle associated with uniform priors assumed

    """
    return -0.5 * nDims / (Z * priorVolume) * \
        (1. + np.log(2. * np.pi)) - np.log(Z)
