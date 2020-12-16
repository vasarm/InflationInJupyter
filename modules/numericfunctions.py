import numpy as np

from scipy.integrate import odeint
from scipy.optimize import newton
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

def find_function_roots_numerical(function, settings, method = 1):

    x_interval = settings.create_interval_list()
    if method == 1:
        dx = settings.root_classify
        roots = find_root_method_one(function, x_interval)
        return roots[(function(roots) < dx) & (function(roots) > -dx)]
    elif method == 2:
        pass

def find_root_method_one(function, x_interval):
    """
    Calcualte given function roots. This method actually calculates
    function values, takes absolute value from every value and then
    looks for minimas (lowest extremas).

    Parameters
    ----------
    function : FunctionType
        Function which roots are looked for
    x_interval : array_like
        x all possible values

    Returns
    -------
    array_like
        All possible roots in defined interval
    """
    return x_interval[argrelextrema(np.absolute(function(x_interval)), np.less)]


def find_root_numerical(function, x0, fprime=None, fprime2=None):
    """
    Calcualte more precise minima value.

    Parameters
    ----------
    function : FunctionType
        Function whcih root is calculated
    x0 : float
        Near that point root is defined
    fprime : [type], optional
        [description], by default None
    fprime2 : [type], optional
        [description], by default None

    Returns
    -------
    float
        funtction root
    """
    return newton(func=function, x0=x0, fprime=fprime, fprime2=fprime2)

# returns value so that x-axis is N_values and y-axis are scalar field initial values
def integrate_N_fold_numerical(function, end_value, N_list):
    """
    Solve differential equation for dφ/dN for given N values.

    Parameters
    ----------
    function : FunctionType
        Differential expression dφ/dN
    end_value : float
        One point to solve differential equation
    N_list : list
        Possible N values list.

    Returns
    -------
    scipy.interp1d
        Function φ(N)
    """
    def integrate_function(y, x):
        return function(y)

    with np.errstate(divide='ignore'):
        y_scalar_field_values = np.squeeze(odeint(func=integrate_function, y0=end_value, t=N_list))

        N_function = interp1d(N_list, y_scalar_field_values)
        return N_function















