import numpy as np

# misc functions


def logAddExp(a, b, fact):
    """
Calculates fact + log(exp(a - fact) + exp(b - fact)) for scalars a and b.
Subtracts fact from a and b before exponentiating
in attempt to avoid underflow when a and b are small
or overflow when a and b are large,
and adds back to final result
"""
    expa = np.exp(a - fact)
    expb = np.exp(b - fact)
    return fact + np.log(expa + expb)


def logAddArr(x, y):
    """
    logaddexp where x is a scalar and y is an array.
    Calculates log(exp(x) + sum(exp(y))).
    Returns a scalar in case that axis = None (exponentiates elements of array then adds them together).
    np implementation on its own returns an array i.e. doesn't sum exponentiated elements of array.
    Axis specifies axis to do summation of exponentials over, and if specified returns an array with shape of the remaining dimensions.
    Doesn't try to avoid under/overflow when logaddexp'ing the array y
    but is more efficient than logAddArr2 as it is vectorised.
    Axis should be specified such that summation over y results in scalar
    probably better to use [scipy.misc.]logsumexp(y) and then logaddexp(x,y), although mirculously, this isn't very fast
    """
    yExp = np.exp(y)
    logyExpSum = np.log(yExp.sum())
    return np.logaddexp(x, logyExpSum)


def logAddArr2(x, y, indexes=(None, )):
    """
    Alternative version of logAddArr that avoids over/ underflow errors of exponentiating the array y, to the same extent that np.logaddexp() does
    Note however that it is slower than logAddArr, so in cases where over/ underflow isn't an issue, use that
    Loops over each specified element of y (using rowcol values) and adds to log of sums.
    Note when trying to avoid under/overflow it only subtracts max(result so far, y_i) for each iteration through y, it doesn't
    subtract max(y) from all iterations
    By default loops over entire array, but to specify a certain row for e.g. a 2d array set indexes to (row_index, slice(None))
    or for a certain column (slice(None), col_index)
    """
    result = x
    for l in np.nditer(y[indexes]):
        result = np.logaddexp(result, l)
    return result


def logAddArr3(x, y, indexes=(None, )):
    """
    as above but subtracts max(x, y) when doing logAddExp
    """
    yMax = np.max(y)
    totMax = max(x, yMax)
    result = x
    for l in np.nditer(y[indexes]):
        result = logAddExp(result, l, totMax)
    return result


def logSubExp(a, b, fact):
    """
    Calculates fact + log(exp(a - fact) - exp(b - fact)) for scalars a and b.
    Subtracts fact from a and b before exponentiating
    in attempt to avoid underflow when a and b are small
    or overflow when a and b are large,
    and adds back to final result
    """
    if b > a:
        raise FloatingPointError(
            'encountered log of a negative number in logSubExp(a,b); a = %s; b = %s'
            % (a, b))
    expa = np.exp(a - fact)
    expb = np.exp(b - fact)
    return fact + np.log(expa - expb)


def logSubArr2(x, y, indexes=(None, )):
    """
    Same as logAddArr2 but with logSubExp
    calculates log(exp(x) - exp(sum(y))).
    Named ...2 for consistency
    """
    result = x
    for l in np.nditer(y[indexes]):
        result = np.logSubExp(result, l, result)
    return result


def logSubArr3(x, y, indexes=(None, )):
    """
    Same as logAddArr3 but with logSubExp
    calculates log(exp(x) - exp(sum(y))).
    Named ...3 for consistency
    """
    yMax = np.max(y)
    totMax = max(x, yMax)
    result = x
    for l in np.nditer(y[indexes]):
        result = logSubExp(result, l, totMax)
    return result
