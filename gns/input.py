# import standard modules
import numpy as np

# import custom modules


def getFromTxt(file):
    """
    get separate arrays of weights, LLhood and params from
    getDist .txt file
    """
    arr = np.genfromtxt(file)
    weights = arr[:, 0]
    LLhood = arr[:, 1]
    params = arr[:, 2:]
    return weights, LLhood, params


def getFromSummary(file):
    """
    get separate arrays of params, Lhood (or LLhood), weights and X values from
    summary.txt file
    """
    arr = np.genfromtxt(file, skip_header=1)
    params = arr[:, :-3]
    Lhood = arr[:, -3]
    weights = arr[:, -2]
    XArr = arr[:, -1]
    return params, Lhood, weights, XArr


def clipChains(chains, clippedChains, n):
    """
    clip chains by selecting every nth row
    chains is chains filename, clippedChains is filename to save clipped data to
    """
    weights, LL, params = getFromTxt(chains)
    clippedWeights, clippedLogLike, clippedParams = weights[::n].reshape(
        -1, 1), LL[::n].reshape(-1, 1), params[::n, :]
    clippedArr = np.hstack((clippedWeights, clippedLogLike, clippedParams))
    np.savetxt(clippedChains, clippedArr)
