import sympy as sp
import numpy as np
import scipy as sc

import types
import inspect

from scipy.misc import derivative

from .. import errors as errors


class InflationFunctionBase:
    def __init__(self, function, **kwargs):
        """
        Base class for all inflation functions. This class
        has basic attributes which all functions have.
        This class defines numeric and symbolic functions.
        Also method to calculate derivatives.
        """
        self.name = ""
        self._symbol = sp.symbol("x", real=True)
        self.symbolic_function = function
        self.numeric_function = function

    @property
    def symbolic_function(self):
        return self.__symbolic_function

    @symbolic_function.setter
    def symbolic_function(self, function):
        """
        Checks if entered function can be transformed to sympy formatted function.
        """

        if isinstance(function, sp.Expr):
            variables = [(x, sp.Symbol(x, real=True)) for x in function.free_symbols]
            function = function.subs(variables)
        elif isinstance(function, str):
            try:
                function = sp.sympify(function)
                variables = [(x, sp.Symbol(x, real=True)) for x in function.free_symbols]
                function = function.subs(variables)
            except:
                raise RuntimeError("Can't convert string to sympy expression.")
        else:
            function = None

        self.__symbolic_function = function


    @property
    def numeric_function(self):
        return self.__numeric_function

    @numeric_function.setter
    def numeric_function(self, function):
        """
        Converst function to python function.
        """
        if isinstance(function, sp.Expr) or isinstance(str):
            variables = self.symbolic_function.free_symbols - set(self._symbol)
            variables = sorted(list(variables), key=str)
            variables.insert(0, self._symbol)
            function = sp.lambdify(variables, self.symbolic_function, "numpy")
        elif isinstance(function, types.FunctionType) or isinstance(function, sc.interpolate.interpolate.interp1d):
            pass
        else:
            raise errors.WrongTypeError("Function must be string/sympy expression/scipy interp1d or FunctionType.")

        self.__numeric_function = function

    def symbolic_function(self):
        """
        Returns symbolic function

        Returns
        -------
        symbolic_function: sp.Expr
            Retruns symbolic function
        """
        return self.symbolic_function

    def numeric_function(self, x, **kwargs):
        """
        Returns function value at point x.

        Parameters
        ----------
        x : array_like
            Points where function value is calculated.
        **kwargs: float
            Function parameters values.

        Returns
        -------
        function_value: array_like
            Returns function value at point x.
        """
        return self.numeric_function(x, **kwargs)

    def symbolic_derivative(self):
        if self.symbolic_function is not None:
            return sp.diff(self.symbolic_function, self.symbol)
        else:
            raise errors.FunctionNotDefinedError("Symbolic function for '{}' is not defined.".format(self.name))

    def symbolic_second_derivative(self):
        if self.symbolic_function is not None:
            return sp.diff(self.symbolic_function, self.symbol, 2)
        else:
            raise errors.FunctionNotDefinedError("Symbolic function for '{}' is not defined.".format(self.name))

    def numeric_derivative(self, x, dx, **kwargs):
        """
        Calculate numerically function's derivative. If this function has defined symbolic function
        then it uses this for mor accuracy. Else calculate numerically.

        Parameters
        ----------
        x : array_like
            Value where derivative is calculated.
        dx : float
            Increment for derivative calculation.
        **kwargs: float
            Function parameter values.

        Returns
        -------
        derivative_value: array_like
            Function derivative in point x.
        """
        if self.symbolic_function is not None:
            symbolic_derivative = self.symbolic_derivative()
            variables = self._variables()
            variables = [sp.Symbol(x, real=True) for x in variables]
            function = sp.lambdify(variables, symbolic_derivative, "numpy")
            return function(x, **kwargs)
        else:
            try:
                function = lambda y: self.numeric_function(x=y, **kwargs)
            except ZeroDivisionError:
                ZeroDivisionError("Division by zero occured.")
            except:
                raise errors.ParameterDefinitionError("Inserted number of parameters and function parameters is not same.")
            return derivative(func=function, x0=x, n=1, dx=dx)

    def numeric_second_derivative(self, x, dx, **kwargs):
        """
        Calculate numerically function's second derivatve. If this function has defined symbolic function
        then it uses this for mor accuracy. Else calculate numerically.

        Parameters
        ----------
        x : array_like
            Value where derivative is calculated.
        dx : float
            Increment for derivative calculation.
        **kwargs: float
            Function parameter values.

        Returns
        -------
        derivative_value: array_like
            Function second derivative in point x.
        """
        if self.symbolic_function is not None:
            symbolic_derivative = self.symbolic_second_derivative()
            variables = self._variables()
            variables = [sp.Symbol(x, real=True) for x in variables]
            function = sp.lambdify(variables, symbolic_derivative, "numpy")
            return function(x, **kwargs)
        else:
            try:
                function = lambda y: self.numeric_function(x=y, **kwargs)
            except ZeroDivisionError:
                ZeroDivisionError("Division by zero occured.")
            except:
                raise errors.ParameterDefinitionError("Inserted number of parameters and function parameters is not same.")
            return derivative(func=function, x0=x, n=2, dx=dx)



def _variables(self):
    return tuple(inspect.getfullargspec(self.numeric_function).args)