# import standard modules
import numpy as np
import scipy.stats

# import custom modules

# convert from phys point isin [l, u] to phi isin [0, 2pi]


def physPeriod2TwoPi(p, l, u):
    return 2. * np.pi * (p - l) / (u - l)


# convert from phi isin [0, 2pi] to p isin [l, u]


def twoPiPeriod2Phys(phi, l, u):
    return phi * (u - l) / (2. * np.pi) + l


# convert from phys point isin [l, u] to phi isin [0, pi]
def physPeriod2Pi(p, l, u):
    return np.pi * (p - l) / (u - l)


# convert from phi isin [0, 2pi] to p isin [l, u]


def piPeriod2Phys(phi, l, u):
    return phi * (u - l) / np.pi + l


# converts from angle, which takes values isin [-pi, pi] to angle which takes values isin [0, 2pi]
# needed because np.arc functions return a value isin [-pi, pi], and has the convention that if the angle is measured from the x-axis,
# then at the following axes it takes values:
# positive x: 0; positive y: pi/2; negative x: pi; negative y: -pi/2
# i.e. it measures the positive angle from the positive x-axis counter clockwise and the negative angle clockwise
# after this conversion it takes the values:
# positive x: 0; positive y: pi/2; negative x: pi; negative y: 3pi/2
# i.e. it measures the positive anti-clockwise angle from the positive x-axis


def switchPolarSys(phi):
    return 2. * np.pi - np.abs(phi) if phi < 0. else phi


def switchTorusSys(theta, R, rho):
    # upper right quadrant of tube cross-section (ACW)
    if rho >= R and theta >= 0.:
        return theta
    # upper left quadrant of tube cross-section (ACW)
    elif rho < R and theta > 0.:
        return np.pi - theta
    # bottom left quadrant of tube cross-section (ACW)
    elif rho < R and theta <= 0.:
        return np.abs(theta) + np.pi
    # bottom right quadrant of tube cross-section (ACW)
    elif rho >= R and theta < 0.:
        return 2. * np.pi - np.abs(theta)


def point2CartCirc(p, l, u):
    """
    Takes a periodic 1-d physical point along with its bounds,
    projects it onto the unit circle (r = 1) and returns its 2-d
    Cartesian coordinates on this circle
    """
    r = 1.
    phi = physPeriod2TwoPi(p, l, u)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def cartCirc2Point(x, y, l, u):
    """
    Convert from Cartesian coordinates back to 1-d angle parameterising circle.
    Then converts back to physical point value based on upper and lower limits
    Note y = 0 gives p = 0 for all x
    """
    phi = np.arctan2(
        y, x
    )  # gives an angle isin [-pi, pi] measured counterclockwise from positive x-axis
    phi = switchPolarSys(phi)
    p = twoPiPeriod2Phys(phi, l, u)
    return p


def projectCart2Circ(x, y):
    """
    Takes arbitrary point in x-y plane
    and projects onto unit circle.
    Does same as first part of cartCirc2Point()
    Returns angle measured from positive x-axis
    """
    phi = np.arctan2(
        y, x
    )  # gives an angle isin [-pi, pi] measured counterclockwise from positive x-axis
    phi = switchPolarSys(phi)
    return phi


def point2CartTorus(p1, p2, l1, l2, u1, u2):
    """
    Takes a periodic 2-d physical point along with its bounds,
    Converts it into a point on a torus which is parameterised by these periodic values and returns its 3-d
    Cartesian coordinates on the torus.
    Torus is defined to have greater radius = 2 and lesser radius = 1
    """
    r = 1.
    R = 2.
    phi = physPeriod2TwoPi(p1, l1, u1)
    theta = physPeriod2TwoPi(p2, l2, u2)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def cartTorus2Point(x, y, z, l1, l2, u1, u2):
    """
    Convert from Cartesian coordinates on torus back to 2-d angle parameterising torus.
    Then converts back to physical point value based on upper and lower limits
    Again assumes r = R = 1
    """
    r = 1.
    R = 2.
    phi = np.arctan2(y, x)
    phi = switchPolarSys(phi)
    p1 = twoPiPeriod2Phys(phi, l1, u1)
    rho = np.sqrt(x**2. + y**2.)
    # gives an angle isin [-pi, pi] measured counterclockwise from axis
    # parallel to x-y plane outwards from centre of tube
    theta = np.arcsin(z / r)
    theta = switchTorusSys(theta, R, rho)
    p2 = twoPiPeriod2Phys(theta, l2, u2)
    return p1, p2


def projectCart2Torus(x, y, z, R):
    """
    Takes arbitrary point in 3-d cartesian coordinates
    and projects it onto torus centred on origin with greater torus
    radius R.
    First calculates azimuthal angle by considering projection of point onto
    circle in x-y plane with radius equal to R.
    Then calculates angle around tube by taking arcsin of ratio of
    z component of original point to distance from
    original point to centre of tube at azimuthal angle.
    Returns phi and theta
    """
    phi = np.arctan2(y, x)
    phi = switchPolarSys(phi)
    # get point on circle in x-y plane which runs along centre of tube of torus
    xCirc = R * np.cos(phi)
    yCirc = R * np.sin(phi)
    # distance from point to nearest point on centre of tube
    tubeDist = np.sqrt((x - xCirc)**2. + (y - yCirc)**2. + z**2.)
    theta = np.arcsin(z / tubeDist)
    rho = np.sqrt(x**2. + y**2.)
    theta = switchTorusSys(theta, R, rho)
    return phi, theta


def point2CartSphere(p1, p2, l1, l2, u1, u2):
    """
    Takes a periodic 2-d physical point along with its bounds,
    projects it onto surface of 'unit sphere' which is parameterised by these periodic values and returns its 3-d
    Cartesian coordinates on the sphere.
    Sphere is defined to have radius 1
    """
    r = 1.
    phi = physPeriod2TwoPi(p1, l1, u1)
    theta = physPeriod2Pi(p2, l2, u2)
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def cartSphere2Point(x, y, z, l1, l2, u1, u2):
    """
    Convert from Cartesian coordinates back to 2-d angle parameterising surface of sphere.
    Then converts back to physical point value based on upper and lower limits
    Again assumes r = 1
    """
    r = 1.
    phi = np.arctan2(y, x)
    phi = switchPolarSys(phi)
    p1 = twoPiPeriod2Phys(phi, l1, u1)
    # gives an angle isin [0, pi] measured counterclockwise from positive
    # z-axis
    theta = np.arccos(z / r)
    p2 = piPeriod2Phys(theta, l2, u2)
    return p1, p2


def projectCart2Sphere(x, y, z):
    """
    Takes arbitrary point in 3-d cartesian coordinates
    and projects it onto sphere centred on origin.
    Essentially does same as first part of cartSphere2Point()
    but for arbitrary radius r.
    Returns phi and theta
    """
    phi = np.arctan2(y, x)
    phi = switchPolarSys(phi)
    r = np.sqrt(x**2. + y**2. + z**2.)
    # gives an angle isin [0, pi] measured counterclockwise from positive
    # z-axis
    theta = np.arccos(z / r)
    return phi, theta


def testCircleProposalSymmetry(p, l, u, cov):
    """
    takes a single point on circle (parameterised by angle)
    and evaluates 2-d gaussian in cartesian coords at a random point
    which is sampled from same 2-d Gaussian, centred on same input point.
    Then projects random point onto circle, and evaluates pdf of a gaussian
    centred on the proposed point at input point.
    Checks if these two pdf values are the same
    Essentially checks if (phi' | phi) = p(phi | phi')
    where p is proposal distribution of Metropolis algo
    """
    x, y = point2CartCirc(p, l, u)
    xy = np.array([x, y])
    xyPrime = np.random.multivariate_normal(mean=xy, cov=cov)
    xyPrimeProb = scipy.stats.multivariate_normal.pdf(x=xyPrime,
                                                      mean=xy,
                                                      cov=cov)
    pPrime2 = cartCirc2Point(xyPrime[0], xyPrime[1], l, u)
    xPrime2, yPrime2 = point2CartCirc(pPrime2, l, u)
    xyPrime2 = np.array([xPrime2, yPrime2])
    xyProb = scipy.stats.multivariate_normal.pdf(x=xy, mean=xyPrime2, cov=cov)
    return xyProb == xyPrimeProb


def testTorusProposalSymmetry(p1, p2, l1, l2, u1, u2, cov):
    """
    takes a single point on torus (parameterised by 2 angles)
    and evaluates 3-d gaussian in cartesian coords at a random point
    which is sampled from same 3-d Gaussian, centred on same input point.
    Then projects random point onto torus, and evaluates pdf of a gaussian
    centred on the proposed point at input point.
    Checks if these two pdf values are the same
    Essentially checks if p(phi', theta' | phi, theta) = p(phi, theta | phi', theta')
    where p is proposal distribution of Metropolis algo
    """
    x, y, z = point2CartTorus(p1, p2, l1, l2, u1, u2)
    xyz = np.array([x, y, z])
    xyzPrime = np.random.multivariate_normal(mean=xyz, cov=cov)
    xyzPrimeProb = scipy.stats.multivariate_normal.pdf(x=xyzPrime,
                                                       mean=xyz,
                                                       cov=cov)
    p1Prime2, p2Prime2 = cartTorus2Point(xyzPrime[0], xyzPrime[1], xyzPrime[2],
                                         l1, l2, u1, u2)
    xPrime2, yPrime2, zPrime2 = point2CartTorus(p1Prime2, p2Prime2, l1, l2, u1,
                                                u2)
    xyzPrime2 = np.array([xPrime2, yPrime2, zPrime2])
    xyzProb = scipy.stats.multivariate_normal.pdf(x=xyz,
                                                  mean=xyzPrime2,
                                                  cov=cov)
    return xyzProb == xyzPrimeProb


def testSphereProposalSymmetry(p1, p2, l1, l2, u1, u2, cov):
    """
    takes a single point on surface of sphere (parameterised by 2 angles)
    and evaluates 3-d gaussian in cartesian coords at a random point
    which is sampled from same 3-d Gaussian, centred on same input point.
    Then projects random point onto sphere, and evaluates pdf of a gaussian
    centred on the proposed point at input point.
    Checks if these two pdf values are the same
    Essentially checks if p(phi', theta' | phi, theta) = p(phi, theta | phi', theta')
    where p is proposal distribution of Metropolis algo
    """
    x, y, z = point2CartSphere(p1, p2, l1, l2, u1, u2)
    xyz = np.array([x, y, z])
    xyzPrime = np.random.multivariate_normal(mean=xyz, cov=cov)
    xyzPrimeProb = scipy.stats.multivariate_normal.pdf(x=xyzPrime,
                                                       mean=xyz,
                                                       cov=cov)
    p1Prime2, p2Prime2 = cartSphere2Point(xyzPrime[0], xyzPrime[1],
                                          xyzPrime[2], l1, l2, u1, u2)
    xPrime2, yPrime2, zPrime2 = point2CartSphere(p1Prime2, p2Prime2, l1, l2,
                                                 u1, u2)
    xyzPrime2 = np.array([xPrime2, yPrime2, zPrime2])
    xyzProb = scipy.stats.multivariate_normal.pdf(x=xyz,
                                                  mean=xyzPrime2,
                                                  cov=cov)
    return xyzProb == xyzPrimeProb


def cartesianFromSpherical(phi, theta):
    """
    get cartesians from sphericals
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def sphericalTranslation(phi0, theta0, dPhi, dTheta):
    """
    phi0 and theta0 correspond to x0, y0, z0
    dphi, dtheta correspond to some dx, dy, dz
    returns coordinates xt, yt, zt
    """
    xt = (np.cos(phi0) * np.cos(dPhi) - np.sin(phi0) * np.sin(dPhi)) * \
        (np.sin(theta0) * np.cos(dTheta) + np.cos(theta0) * np.sin(dTheta))
    yt = (np.sin(phi0) * np.cos(dPhi) + np.cos(phi0) * np.sin(dPhi)) * \
        (np.sin(theta0) * np.cos(dTheta) + np.cos(theta0) * np.sin(dTheta))
    zt = np.cos(theta0) * np.cos(dTheta) - np.sin(theta0) * np.sin(dTheta)
    return xt, yt, zt


def translateAxis(v, dPhi, dTheta):
    """
    Translates input (Cartesian) vector by dphi, dtheta, returns
    translated (Cartesian) vector
    """
    x0, y0, z0 = v[0], v[1], v[2]
    phi0, theta0 = projectCart2Sphere(x0, y0, z0)
    return np.array(sphericalTranslation(phi0, theta0, dPhi, dTheta))


def translateAxes(axes, dPhi, dTheta):
    """calls translateAxis on each vector in axes, and shifts each by dPhi, dTheta"""
    translatedAxes = []
    for v in axes:
        translatedAxes.append(translateAxis(v, dPhi, dTheta))
    return translatedAxes


def testOrthog(vs):
    n = len(vs)
    tol = 0.01
    dots = []
    orthList = []
    for i in range(n):
        for j in range(i, n):
            dots.append(np.dot(vs[i], vs[j]))
            if i == j:
                orthList.append(1.)
            else:
                orthList.append(0.)
    return dots, np.all(np.abs(np.array(orthList) - np.array(dots)) < tol)


def getRandOrthogs(v1):
    """
    get two vectors (mutually) orthogonal to v1.
    ASSUMES v1 is normalised
    """
    v2 = np.random.randn(3)
    v2 -= np.dot(v1, v2) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return v2, v3


def rodriguezRot(k, v, alpha):
    """
    Rotate vector v about angle alpha (defined b y right hand rule) in plane defined by unit vector k (plane perpendicular to k).
    """
    return v * np.cos(alpha) + np.cross(k, v) * np.sin(alpha) + \
        k * np.dot(k, v) * (1 - np.cos(alpha))


def getPoleVec(v):
    """
    takes vector v, which is normal to plane tangential to surface of unit sphere (i.e. unit vector from origin), and finds vector in perpendicular plane which points to (positive) z axis. Since v is also a point on the perpendicular plane, only need single vector to get pole vector from n_hat dot (x - a) = 0 i.e. n_hat = a = v.
    ASSUMES v is normalised
    """
    z = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / v[2]
    poleVec = np.array([0., 0., z]) - v
    poleVec /= np.linalg.norm(poleVec)
    return poleVec


def getPoleOrthogs(v1):
    """
    get vector orthogonal to v1, which points towards positive z-pole, then finds vector orthogonal to v1 and
    v2 by using cross product
    """
    v2 = getPoleVec(v1)
    v3 = np.cross(v1, v2)
    return v2, v3


if __name__ == '__main__':
    pass
