import numpy as np
import sympy as sp
from sympy.solvers import solve

import multiprocessing
from queue import Empty

from . import errors as errors

def subprocess(func, *args, time=10):
    """
    Function which calls out defined functions and terminates them after time.
    Becaues of that it creates another thread (using module multiprocessing).

    Parameters
    ----------
    func : FunctionType
        Defined function to call out
    time : int, optional
        Time in seconds after what thread is terminated, by default 10

    Returns
    -------
    Any (string, float, sp.Expr, )
        Solution

    Raises
    ------
    TimeoutError
        After define time if program is terminated.
    Errors.NoSolutionError
        Couldn't calculate solution for function
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=func, args=(*args, queue, ))
    process.start()
    process.join(time + 1)

    if process.is_alive():
        process.terminate()
        raise TimeoutError("Process was terminated because it took too much time.")
    elif queue.empty():
        raise errors.NoSolutionError("Error. Could not find any solution.")
    else:
        try:
            return queue.get()
        except Empty:
            raise errors.NoSolutionError("Error. Could not find any solution.")

def calculate_scalar_field_end_value(epsilon, queue=None):
    """
    Calculates scalar field value symbolically at the end of the inflation.
    Basically this function looks for function roots.

    Parameters
    ----------
    epsilon : str
        Epsilon function
    queue : [type], optional
        Through Queue the value is returned after process, by default None

    Returns
    -------
    [type]
        [description]
    """
    symbol = sp.Symbol("x", real=True)
    epsilon = sp.sympify(epsilon, {"x": symbol})
    try:
        end_values = list(solve(epsilon - 1, symbol))
        if queue is not None:
            queue.put(end_values)
        else:
            return end_values
    except:
        raise errors.NoSolutionError("Error when solving. Was looking for scalar field end value.")

def run_end_value_calculation_symbolical(epsilon, time):
    """
    Run end value calculation through subprocess.

    Parameters
    ----------
    epsilon : sp.Expr
        Epsilon function
    time : int
        Timeout time

    Returns
    -------
    list
        List of possible solutions
    """
    epsilon = str(epsilon)
    results = subprocess(calculate_scalar_field_end_value, epsilon, time=time)
    # select only real values
    return [np.float(x) for x in results if x.is_real]

def integrate_N_integrand(N_integrand, end_value, symbol, queue=None):
    """
    Integrate to find N(φ) function and then find it's inverse.

    Parameters
    ----------
    N_integrand : string
        Function to be integrated
    end_value : float
        scalar field end value
    queue : Queue, optional
        Through Queue the value is returned after process, by default None

    Returns
    -------
    [type]
        [description]
    """
    symbol = sp.Symbol("x", real=True)
    N_symbol = sp.Symbol("N", real=True, positive=True)
    N_integrand = sp.sympify(N_integrand, {"x": symbol})
    # Integrating function
    # 1) function variable
    # 2) and 3) are lower and upper boundaries
    N_equation = sp.integrate(N_integrand, (symbol, end_value, symbol))
    N_integral = sp.Eq(N_symbol, N_equation)
    try:
        solve_for_scalar_field = solve(N_integral, symbol)
        # Not interested in imaginary solutions so taking these out
        solve_for_scalar_field = tuple([str(x) for x in solve_for_scalar_field if not x.has(sp.I)])
        if queue is not None:
            queue.put(solve_for_scalar_field)
    except:
        raise errors.NoSolutionError("Error when solving. Was integrating and looking for inverse.")

    else:
        return N_equation

def run_N_fold_integration_symbolic(N_integrand, end_value, time=None):
    """
    [summary]

    Parameters
    ----------
    N_integrand : sp.Expr
        Function to be integrated
    end_value : float
        Scalar field value in the end of inflation
    time : [type], optional
        Timeout time, by default None

    Returns
    -------
    list
        List of possible solutions
    """
    N_integrand = str(N_integrand)
    return subprocess(integrate_N_integrand, N_integrand, end_value, time=time)

#
#
#
# Invariants
#
#
#
