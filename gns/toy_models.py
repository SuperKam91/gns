# import standard modules
import numpy as np

# import custom modules
from . import prob_funcs
from . import geom


def getToyHypersGeom(shape):
    """
    wrapper around getToyHypers() to get priors/ lhoods
    specific to geometric nested sampling project.
    NOTE 'sphere' changed to l8 on 18th Feb 2018.
    See getToyHypers for priors/likelihood that these different shapes (models) correspond to.

    Args:

    shape: string representing toy model name e.g. 'torus III'

    Returns:

    values of respective dictionaries, which contain lists of parameter names, prior and likelihood types, and their hyperparameters in the form:

    paramNames: list of strings representing parameter names

    priorParams: list containing prior type (denoted by integer) and hyperparameters (array)

    LhoodParams: list containing likelihood type (denoted by integer) and hyperparameters (array)

    Priors which prior types integers correspond to can be found in fitPriors() in prob_funcs.py.

    Likelihoods which likelihood types integers correspond to can be found in fitLhood() in prob_funcs.py.

    """
    shapeDict = {
        'circle': [
            'n1', 'p1', 'l4'], 'torus': [
            'n2', 'p2', 'l5'], 'torus II': [
                'n4', 'p7', 'l9'], 'torus III': [
                    'n5', 'p8', 'l10'], 'torus IV': [
                        'n11', 'p16', 'l21'], 'torus V': [
                            'n12', 'p17', 'l22'], 'sphere': [
                                'n2', 'p3', 'l8'], 'sphere II': [
                                    'n2', 'p3', 'l7'], 'sphere III': [
                                        'n2', 'p9', 'l11'], 'sphere IV': [
                                            'n2', 'p9', 'l12'], 'sphere V': [
                                                'n2', 'p3', 'l11'], 'sphere VI': [
                                                    'n2', 'p3', 'l12'], '3 sphere IV': [
                                                        'n6', 'p10', 'l13'], '5 sphere IV': [
                                                            'n7', 'p11', 'l14'], '6 sphere IV': [
                                                                'n8', 'p12', 'l15'], '10d gauss sphere IV': [
                                                                    'n9', 'p13', 'l16'], '20d gauss sphere IV': [
                                                                        'n10', 'p14', 'l17'], 'sphere VII': [
                                                                            'n2', 'p9', 'l18'], '20d gauss sphere VII': [
                                                                                'n10', 'p14', 'l19'], '6 sphere VIII': [
                                                                                    'n8', 'p15', 'l20']}
    namesPriorLhood = shapeDict[shape]
    return getToyHypers(*namesPriorLhood)


def getToyHypersGen(dists):
    """
    wrapper around getToyHypers() to get priors/ lhoods
    specific to geometric nested sampling project

    See getToyHypers for priors/likelihood that these different models correspond to.

    Args:

    shape: string representing toy model name e.g. 'gauss p gauss l'

    Returns:

    values of respective dictionaries, which contain lists of parameter names, prior and likelihood types, and their hyperparameters in the form:

    paramNames: list of strings representing parameter names

    priorParams: list containing prior type (denoted by integer) and hyperparameters (array)

    LhoodParams: list containing likelihood type (denoted by integer) and hyperparameters (array)

    Priors which prior types integers correspond to can be found in fitPriors() in prob_funcs.py.

    Likelihoods which likelihood types integers correspond to can be found in fitLhood() in prob_funcs.py.

    """
    genDict = {
        'uniform p gauss l': [
            'n1', 'p1', 'l1'], 'uniform uniform p gauss l': [
            'n2', 'p2', 'l2'], 'gauss p gauss l': [
                'n1', 'p6', 'l1']}
    namesPriorLhood = genDict[dists]
    return getToyHypers(*namesPriorLhood)


def getToyHypers(n, p, l):
    """
    Takes three string arguments which are used in dictionary to look up
    types of priors/ lhoods (denoted by integers) as well as their hyperparameters (arrays).
    The priors/likelihoods that each dictionary entry corresponds to is written above the respective
    dictionary value definition.

    Args:

    n: string for dictionary key of parameter names dictionary e.g. 'n1'

    p: string for dictionary key of priors dictionary e.g. 'p1'

    l: string for dictionary key of likelihoods dictionary e.g. 'l1'

    Returns:

    values of respective dictionaries, which contain lists of parameter names, prior and likelihood types, and their hyperparameters in the form:

    paramNames: list of strings representing parameter names

    priorParams: list containing prior type (denoted by integer) and hyperparameters (array)

    LhoodParams: list containing likelihood type (denoted by integer) and hyperparameters (array)

    Priors which prior types integers correspond to can be found in fitPriors() in prob_funcs.py.

    Likelihoods which likelihood types integers correspond to can be found in fitLhood() in prob_funcs.py.


    """
    # got rid of $ signs so that works with getdist gui without having to change .paramnames
    # one param
    n1 = ['\\phi']
    # two params
    n2 = ['\\phi', '\\theta']
    # three params
    n3 = ['\\phi', '\\theta', '\\rho']
    # four params
    n4 = ['\\theta_{1}', '\\theta_{2}', '\\theta_{3}', '\\theta_{4}']
    # six params
    n5 = n4 + ['\\theta_{5}', '\\theta_{6}']
    # three pairs of spherical params
    n6 = [
        '\\phi_{1}',
        '\\theta_{1}',
        '\\phi_{2}',
        '\\theta_{2}',
        '\\phi_{3}',
        '\\theta_{3}']
    # five pairs of spherical params
    n7 = n6 + ['\\phi_{4}', '\\theta_{4}', '\\phi_{5}', '\\theta_{5}']
    # six pairs of spherical params
    n8 = n7 + ['\\phi_{6}', '\\theta_{6}']
    # 10 params and 2 spherical
    n9 = n5 + ['\\theta_{7}', '\\theta_{8}',
               '\\theta_{9}', '\\theta_{10}'] + n2
    # 20 params and 2 spherical
    n10 = ['\\theta_{i}' for i in range(1, 21)] + n2
    # 8 params
    n11 = n5 + ['\\theta_{7}', '\\theta_{8}']
    # 10 params
    n12 = n11 + ['\\theta_{9}', '\\theta_{10}']
    # uniform on [0, 2pi] (circle)
    p1 = np.array([[1, 0., 2. * np.pi]]).T
    # uniform on [0, 2pi]^2 (torus)
    p2 = np.array([[1, 0., 2. * np.pi], [1, 0., 2. * np.pi]]).T
    # uniform on [0, 2pi] x [0, pi] (sphere)
    p3 = np.array([[1, 0., 2. * np.pi], [1, 0., np.pi]]).T
    # gaussian with mu = 0 and std dev = 1
    p4 = np.array([[2, np.pi, 1.]]).T
    # gaussian with mu = pi and std dev = 1 x uniform on [0, 2pi]
    p5 = np.array([[2, np.pi, 1.], [1, 0., 2. * np.pi]]).T
    # gaussian with mu = 2pi and std dev = 1
    p6 = np.array([[2, 2. * np.pi, 1.]]).T
    # uniform on [0, 2pi]^4 (four-torus)
    p7 = np.array([[1, 0., 2. * np.pi]] * 4).T
    # uniform on [0, 2pi]^6 (six-torus)
    p8 = np.array([[1, 0., 2. * np.pi]] * 6).T
    # uniform on [0, 2pi] and sin[0, pi] (sphere prior II)
    # param vecs irrelevant for sin prior, but used to get support
    p9 = np.array([[1, 0., 2. * np.pi], [3, 0., np.pi]]).T
    # uniform on [0, 2pi] and sin[0, pi]^3 (three spheres prior II)
    p10 = np.array([[1, 0., 2. * np.pi], [3, 0., np.pi]] * 3).T
    # uniform on [0, 2pi] and sin[0, pi] ^5 (five spheres prior II)
    p11 = np.array([[1, 0., 2. * np.pi], [3, 0., np.pi]] * 5).T
    # uniform on [0, 2pi] and sin[0, pi] ^6 (six spheres prior II)
    p12 = np.array([[1, 0., 2. * np.pi], [3, 0., np.pi]] * 6).T
    # uniform on [-3, 3]^10 x [0, 2pi] and sin[0, pi] (10d gauss one sphere
    # prior II)
    p13 = np.array([[1, -3., 3.]] * 10 +
                   [[1, 0., 2. * np.pi], [3, 0., np.pi]]).T
    # uniform on [-3, 3]^20 x [0, 2pi] and sin[0, pi] (20d gauss one sphere
    # prior II)
    p14 = np.array([[1, -3., 3.]] * 20 +
                   [[1, 0., 2. * np.pi], [3, 0., np.pi]]).T
    # uniform on [0, 2pi] and [0, pi] ^6 (six spheres prior I)
    p15 = np.array([[1, 0., 2. * np.pi], [1, 0., np.pi]] * 6).T
    # uniform on [0, 2pi]^8 (eight-torus)
    p16 = np.array([[1, 0., 2. * np.pi]] * 8).T
    # uniform on [0, 2pi]^10 (ten-torus)
    p17 = np.array([[1, 0., 2. * np.pi]] * 10).T
    # gaussian with mu = pi and cov = 1
    l1 = [2, np.array([np.pi]).reshape(1, 1), np.array([1.]).reshape(1, 1)]
    # gaussian with mu = (pi,pi) and cov = (1,0,0,1)
    l2 = [2, np.array([np.pi, np.pi]).reshape(
        1, 2), np.array([1., 0., 0., 1.]).reshape(2, 2)]
    # gaussian with mu = (pi,pi,pi) and cov = (1,0,0,0,1,0,0,0,1)
    l3 = [2, np.array([np.pi, np.pi, np.pi]).reshape(1, 3), np.array(
        [1., 0., 0., 0., 1., 0., 0., 0., 1.]).reshape(3, 3)]
    # von mises on [0, 2pi] with mu = 0 and var = 1 / kappa = 0.25 (circle)
    l4 = [3, np.array([0.]).reshape(1, 1), np.array([0.25]).reshape(1, 1)]
    # von mises on [0, 2pi]^2 with mu = (0,0) and var = 1 / kappa =
    # (0.25,0.25) (torus)
    l5 = [4, np.array([0., 0.]).reshape(1, 2), np.array(
        [0.25, 0., 0., 0.25]).reshape(2, 2)]
    # uniform on [0, 2pi] x truncated gaussian on [0, pi] with mu = (0,0)  and
    # std dev = (.,0.5) (sphere) std dev reduced from 2 to 0.5 on 21/01/18
    l6 = [5, np.array([0., 0.]).reshape(1, 2), np.array(
        [0., 0., 0., 0.5]).reshape(2, 2)]
    # von mises on [0, 2pi] x truncated gaussian on [0, pi] with mu = (0,
    # pi/2), for the former var = 1 / kappa = 0.25, for latter std dev = 0.5
    # (sphere II)
    l7 = [6, np.array([0., np.pi / 2.]).reshape(1, 2),
          np.array([0.25, 0., 0., 0.5]).reshape(2, 2)]
    # von mises on [0, 2pi] x truncated gaussian on [0, pi] with mu = (0, 0),
    # for the former var = 1 / kappa = 0.25, for latter std dev = 1 (sphere)
    l8 = [7, np.array([0., 0.]).reshape(1, 2), np.array(
        [0.25, 0., 0., 1]).reshape(2, 2)]
    # von mises on [0, 2pi]^4 with mu = (0,0,0,0) and var = 1 / kappa =
    # (0.25,0.25,0.25,0.25) (four-torus)
    l9 = [8, np.array([0.] * 4).reshape(1, 4), np.diag([0.25] * 4)]
    # von mises on [0, 2pi]^6 with mu = (0,0,0,0,0,0) and var = 1 / kappa =
    # (0.25,0.25,0.25,0.25,0.25,0.25) (six-torus)
    l10 = [9, np.array([0.] * 6).reshape(1, 6), np.diag([0.25] * 6)]
    # CHANGED TO UNIT VARIANCE KJ 9TH MAY 2019 (suffixed _ds)
    # von mises on [0, 2pi]^6 with mu = (0,0,0,0,0,0) and var = 1 / kappa = (1.,1.,1.,1.,1.,1.) (six-torus)
    # l10 = [9, np.array([0.] * 6).reshape(1,6), np.diag([1.] * 6)]
    # kent distribution with gamma1 = g1, gamma2 = g2, gamma3 = g3, kappa = k,
    # beta = b
    g1 = np.array([0, 0, 1])
    g2 = np.array([0, 1, 0])
    g3 = np.array([1, 0, 0])
    G = np.concatenate([g1, g2, g3])
    k = 100.
    b = 0.5 * k
    l11 = [10, G, np.array([k, b])]
    # kent distribution sum which resembles "star" shape centred on 0,0,1
    k1 = 100.
    b1 = k1 * 0.5
    g11 = np.array([0, 0, 1])
    g12 = np.array([0, 1, 0])
    g13 = np.array([1, 0, 0])
    k2 = 100.
    b2 = k2 * 0.5
    g21 = np.array([0, 0, 1])
    g22 = np.array([1, 0, 0])
    g23 = np.array([0, 1, 0])
    k3 = 100.
    b3 = k3 * 0.5
    g31 = np.array([0, 0, 1])
    g32 = 1. / np.sqrt(2) * np.array([-1, 1, 0])
    g33 = 1. / np.sqrt(2) * np.array([1, 1, 0])
    k4 = 100.
    b4 = k4 * 0.5
    g41 = np.array([0, 0, 1])
    g42 = 1. / np.sqrt(2) * np.array([1, 1, 0])
    g43 = 1. / np.sqrt(2) * np.array([-1, 1, 0])
    g1s = [g11, g21, g31, g41]
    g2s = [g12, g22, g32, g42]
    g3s = [g13, g23, g33, g43]
    kappas = [k1, k2, k3, k4]
    betas = [b1, b2, b3, b4]
    l12 = [11, [g1s, g2s, g3s], [kappas, betas]]
    # three kent star-shaped distributions defined on separate spheres
    l13 = [12, [g1s, g2s, g3s] * 3, [kappas, betas] * 3]
    # five kent star-shaped distributions defined on separate spheres
    l14 = [13, [g1s, g2s, g3s] * 5, [kappas, betas] * 5]
    # six kent star-shaped distributions defined on separate spheres
    l15 = [14, [g1s, g2s, g3s] * 6, [kappas, betas] * 6]
    # kent star-shaped distribution and 10d spherical Gaussian.
    # note in previous likelihoods which use scipystats, mu has been shaped (1, nDims) for consistency with one another,
    # but here we shape it (nDims,), so that it doesn't have to be reshaped before being passed to multivariate norm
    # (I think all scipy stats funcs require form (nDims,), so don't know why I did (1, nDims) in first place)
    gaussMu = np.array([0.] * 10)
    gaussVar = np.diag([1.] * 10)
    l16 = [15, [gaussMu, g1s, g2s, g3s], [gaussVar, kappas, betas]]
    # kent star-shaped distribution and 20d spherical Gaussian.
    gaussMu = np.array([0.] * 20)
    gaussVar = np.diag([1.] * 20)
    l17 = [15, [gaussMu, g1s, g2s, g3s], [gaussVar, kappas, betas]]
    # kent extended star-shaped distribution #skips 5-8 because of naming
    # convention in spherical_kde/kent.py
    g91 = np.array([0., 1. / np.sqrt(2), 1. / np.sqrt(2)])
    g91 = geom.cartesianFromSpherical(np.pi / 2., np.pi / 15)
    g92, g93 = geom.getPoleOrthogs(g91)
    g101 = np.array([-1. / np.sqrt(2), 0., 1. / np.sqrt(2)])
    g101 = geom.cartesianFromSpherical(np.pi, np.pi / 15)
    g102, g103 = geom.getPoleOrthogs(g101)
    g111 = np.array([0., -1. / np.sqrt(2), 1. / np.sqrt(2)])
    g111 = geom.cartesianFromSpherical(3 * np.pi / 2., np.pi / 15)
    g112, g113 = geom.getPoleOrthogs(g111)
    g121 = np.array([1. / np.sqrt(2), 0., 1. / np.sqrt(2)])
    g121 = geom.cartesianFromSpherical(0., np.pi / 15)
    g122, g123 = geom.getPoleOrthogs(g121)
    g131 = np.array([-0.5, 0.5, 1. / np.sqrt(2)])
    g131 = geom.cartesianFromSpherical(3. * np.pi / 4, np.pi / 15)
    g132, g133 = geom.getPoleOrthogs(g131)
    g141 = np.array([-0.5, -0.5, 1. / np.sqrt(2)])
    g141 = geom.cartesianFromSpherical(5. * np.pi / 4, np.pi / 15)
    g142, g143 = geom.getPoleOrthogs(g141)
    g151 = np.array([0.5, -0.5, 1. / np.sqrt(2)])
    g151 = geom.cartesianFromSpherical(7. * np.pi / 4, np.pi / 15)
    g152, g153 = geom.getPoleOrthogs(g151)
    g161 = np.array([0.5, 0.5, 1. / np.sqrt(2)])
    g161 = geom.cartesianFromSpherical(np.pi / 4, np.pi / 15)
    g162, g163 = geom.getPoleOrthogs(g161)
    g1s = [g11, g21, g31, g41, g91, g101, g111, g121, g131, g141, g151, g161]
    g2s = [g12, g22, g32, g42, g92, g102, g112, g122, g132, g142, g152, g162]
    g3s = [g13, g23, g33, g43, g93, g103, g113, g123, g133, g143, g153, g163]
    kappas = [350.] * 12
    betas = [0.5 * 350.] * 12
    l18 = [11, [g1s, g2s, g3s], [kappas, betas]]
    # kent extended star-shaped distribution and 20d spherical Gaussian.
    l19 = [15, [gaussMu, g1s, g2s, g3s], [gaussVar, kappas, betas]]
    # 6 spheres kent extended petal (through phi = 3pi / 4 and 7pi / 4)
    g1s = [g41, g141, g161]
    g2s = [g42, g142, g162]
    g3s = [g43, g143, g163]
    kappas = [350.] * 3
    betas = [0.5 * 350.] * 3
    l20 = [14, [g1s, g2s, g3s] * 6, [kappas, betas] * 6]
    # von mises on [0, 2pi]^8 with mu = (0,0,0,0,0,0,0,0) and var = 1 / kappa
    # = (0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25) (eight-torus)
    l21 = [16, np.array([0.] * 8).reshape(1, 8), np.diag([0.25] * 8)]
    # CHANGED TO UNIT VARIANCE KJ 9TH MAY 2019 (suffixed _ds)
    # von mises on [0, 2pi]^8 with mu = (0,0,0,0,0,0,0,0) and var = 1 / kappa
    # = (1,1,1,1,1,1,1,1) (eight-torus)
    l21 = [16, np.array([0.] * 8).reshape(1, 8), np.diag([1.] * 8)]
    # von mises on [0, 2pi]^10 with mu = (0,0,0,0,0,0,0,0,0,0) and var = 1 /
    # kappa = (0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25) (ten-torus)
    l22 = [17, np.array([0.] * 10).reshape(1, 10), np.diag([0.25] * 10)]
    # CHANGED TO UNIT VARIANCE KJ 9TH MAY 2019 (suffixed _ds)
    # von mises on [0, 2pi]^10 with mu = (0,0,0,0,0,0,0,0,0,0) and var = 1 /
    # kappa = (1,1,1,1,1,1,1,1,1,1) (ten-torus)
    l22 = [17, np.array([0.] * 10).reshape(1, 10), np.diag([1.] * 10)]
    paramNamesDict = {
        'n1': n1,
        'n2': n2,
        'n3': n3,
        'n4': n4,
        'n5': n5,
        'n6': n6,
        'n7': n7,
        'n8': n8,
        'n9': n9,
        'n10': n10,
        'n11': n11,
        'n12': n12}
    priorParamsDict = {
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'p4': p4,
        'p5': p5,
        'p6': p6,
        'p7': p7,
        'p8': p8,
        'p9': p9,
        'p10': p10,
        'p11': p11,
        'p12': p12,
        'p13': p13,
        'p14': p14,
        'p15': p15,
        'p16': p16,
        'p17': p17}
    LhoodParamsDict = {
        'l1': l1,
        'l2': l2,
        'l3': l3,
        'l4': l4,
        'l5': l5,
        'l6': l6,
        'l7': l7,
        'l8': l8,
        'l9': l9,
        'l10': l10,
        'l11': l11,
        'l12': l12,
        'l13': l13,
        'l14': l14,
        'l15': l15,
        'l16': l16,
        'l17': l17,
        'l18': l18,
        'l19': l19,
        'l20': l20,
        'l21': l21,
        'l22': l22}
    paramNames = paramNamesDict[n]
    priorParams = priorParamsDict[p]
    LhoodParams = LhoodParamsDict[l]
    return paramNames, priorParams, LhoodParams


def getToyObjects(priorParams, LhoodParams):
    """
    fit priors and Lhoods of toy model.

    Args:

    priorParams: list containing prior type (denoted by integer) and hyperparameters (array)

    LhoodParams: list containing likelihood type (denoted by integer) and hyperparameters (array)

    Returns:

    Prior and likelihood objects to be passed to getToyProbFuncs()
    """
    priorObjs = prob_funcs.fitPriors(priorParams)
    LhoodObj = prob_funcs.fitLhood(LhoodParams)
    return priorObjs, LhoodObj


def getToyProbFuncs(priorObjs, LhoodObj):
    """
    Obtain pdf and ppf methods of priors and pdf & logpdf methods of likelihood.

    Args:

    priorObjs: list containing prior objects obtained from getToyObjects()

    LhoodObj: likelihood object obtained from getToyObjects()

    Returns:

    Prior and likelihood probability functions in order: prior, log prior, prior quantile,
    likehood, loglikelihood.

    """
    priorFuncsPdf = prob_funcs.getPriorPdfs(priorObjs)
    priorFuncsLogPdf = prob_funcs.getPriorLogPdfs(priorObjs)
    priorFuncsPpf = prob_funcs.getPriorPpfs(priorObjs)
    LhoodFunc = prob_funcs.Lhood(LhoodObj)
    LLhoodFunc = prob_funcs.LLhood(LhoodObj)
    return priorFuncsPdf, priorFuncsLogPdf, priorFuncsPpf, LhoodFunc, LLhoodFunc


def getToyFuncs(priorParams, LhoodParams):
    """
    get prior and lhood functions which can be passed to NestedRun or multinest wrapper functions,
    i.e. calls the functions getToyObjects() and getToyProbFuncs().

    Args:

    priorParams: list containing prior type (denoted by integer) and hyperparameters (array)

    LhoodParams: list containing likelihood type (denoted by integer) and hyperparameters (array)

    Priors which prior types integers correspond to can be found in fitPriors() in prob_funcs.py.

    Likelihoods which likelihood types integers correspond to can be found in fitLhood() in prob_funcs.py.

    Returns:

    Prior and likelihood probability functions in order: prior, log prior, prior quantile,
    likehood, loglikelihood.

    """
    priorObjs, LhoodObj = getToyObjects(priorParams, LhoodParams)
    priorFuncsPdf, priorFuncsLogPdf, priorFuncsPpf, LhoodFunc, LLhoodFunc = getToyProbFuncs(
        priorObjs, LhoodObj)
    priorObjects = prob_funcs.priorObjs(
        priorFuncsPdf, priorFuncsLogPdf, priorFuncsPpf)
    priorFunc = priorObjects.priorFuncsProd
    logPriorFunc = priorObjects.logPriorFuncsSum
    invPriorFunc = priorObjects.invPrior
    return priorFunc, logPriorFunc, invPriorFunc, LhoodFunc, LLhoodFunc


def getTargetSupport(priorParams):
    """
    Returns prior domain for prior specified by priorParams.
    Assumes support of priors (in each dimension) is well connected,
    which it will always be for simple priors considered so far.
    Returns array of shape (3, nDims) where first column is lower bound
    on prior, second is upper bound, and third is difference between too
    (note inf - -inf is set equal to inf).
    Fourth row is only important for theoretical Z/ H functions, as it tells them
    if the prior corresponding to the dimension is rectangular, and thus if it needs to be
    integrated over or not

    Args:

    priorParams: list containing prior type (denoted by integer) and hyperparameters (array).

    Returns:

    array of target support values in array of shape (3, nDims)
    """
    priorType = priorParams[0, :]
    nDims = len(priorType)
    param1Vec = priorParams[1, :]
    param2Vec = priorParams[2, :]
    targetSupport = np.zeros_like(priorParams)
    for i in range(nDims):
        if priorType[i] == 1:
            targetSupport[0, i] = param1Vec[i]
            targetSupport[1, i] = param2Vec[i]
            targetSupport[2, i] = np.abs(
                targetSupport[1, i] - targetSupport[0, i])
        elif priorType[i] == 2:
            targetSupport[0, i] = - np.inf
            targetSupport[1, i] = np.inf
            targetSupport[2, i] = np.inf
        elif priorType[i] == 3:
            targetSupport[0, i] = param1Vec[i]
            targetSupport[1, i] = param2Vec[i]
            targetSupport[2, i] = np.abs(
                targetSupport[1, i] - targetSupport[0, i])
        else:
            print("prior type not recognised")
            sys.exit(1)
    return targetSupport
