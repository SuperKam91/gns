# import standard modules
import numpy as np

# import custom modules


def checkInputParamsShape(priorParams, LhoodParams, nDims):
    """
    checks prior and Lhood input arrays for toy models are correct shape
    """
    assert (priorParams.shape == (3, nDims)
            ), "Prior parameter array should have shape (3, nDims)"
    if LhoodParams[0] < 11:  # skip for kent sums or it's too complicated
        assert (LhoodParams[1].shape == (1, nDims) or LhoodParams[1].shape == (
            9,)), "Llhood params mean array should have shape (1, nDims) or (9,) for Kent distriubtion"
        assert (LhoodParams[2].shape == (nDims, nDims) or LhoodParams[2].shape == (
            2,)), "LLhood covariance array should have shape (nDims, nDims) or be (2,) for Kent distribution"


def checkinvPriorShape(livePointsPhys, livePointsShape):
    """
    check livePointsPhys is correct shape
    """
    assert (livePointsPhys.shape ==
            livePointsShape), "livePointsPhys shape must be same as livePoints shape"


def checkLhoodShape(livePointsLhood, nLive):
    """
    scipy.stats.continuous_rv methods return Lhoods of shape (nLive, 1)
    whereas scipy.stats.multivariate_normal method return Lhoods of shape (nLive,).
    The former is converted to the latter for consistency (and the Keeton equations don't work otherwise)
    """
    assert (livePointsLhood.shape == (nLive, 1) or livePointsLhood.shape == (
        nLive,)), "livePointsLhood wrong shape. should be (%s,) or (%s,1)" % (nLive, nLive)
    if livePointsLhood.shape == (nLive, 1):
        print("converting shape from (%s, 1) to (%s,)" % (nLive, nLive))
        return livePointsLhood.reshape(-1,)
    else:
        return livePointsLhood


def checkTargSupShape(targetSupport, nDims):
    """
    check targetSupport is correct shape
    """
    assert (targetSupport.shape == (3, nDims)
            ), "target support array should have shape (3, nDims)"
