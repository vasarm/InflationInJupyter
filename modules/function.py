import sympy as sp
import numpy as np
import scipy as sc
from sympy import UnevaluatedExpr

import pprint

import types
import inspect

from scipy.misc import derivative

import modules.errors as errors
import modules.config as config


def symbolic_to_numeric(function):
    """
    This function is used to convert symbolic function to numeric function.
    It is required to 'fix' symbol functions which are not simplified.
    This helps to overcome cases where sympy won't simplify cases like (1-x)/(1-x) and it is doing this
    by adding very small increment (something like 10^-100) to divisor. The only problem is that
    real division by zero returns now huge number.
    """
    if not function.args:
        return function

    """
    Check if it's power. In case of being power function then it is required to add small increment to the base.
    Increment is decided so: If there is no other number then add 1e-100 otherwise we need the closest power of
    ten of additives and add -14 + power of ten because otherwise this additives is "lost".
    """
    if isinstance(function, sp.Pow):
        if isinstance(function.args[0], sp.Add) and any(number.is_number for number in function.args[0].args):
            all_numbers = [x for x in function.args[0].args if x.is_number]
            all_number_sum = sum(all_numbers)
            if all_number_sum != 0:
                power = np.floor(np.log10(float(np.abs(sum(all_numbers)))))
            else:
                power = 0
            increment = 10 ** (-14 + power)
        else:
            # If no numbers, then put 10^-100
            increment = 10 ** (-100)
        function = sp.Pow(symbolic_to_numeric(function.args[0]) + increment, function.args[1], evaluate=False)
        return function
    else:
        replacements = []
        for argument in function.args:
            replace = symbolic_to_numeric(argument)
            replacements.append(replace)

        return type(function)(*replacements)


class InflationFunctionBase:
    def __init__(self, function, **kwargs):
        """
        Base class for all inflation functions. This class
        has basic attributes which all functions have.
        This class defines numeric and symbolic functions.
        Also method to calculate derivatives.
        """
        self._symbol = sp.Symbol("x", real=True)
        self.symbolic_function = function
        self.numeric_function = function
        self.name = "function"

    @property
    def symbolic_function(self):
        return self.__symbolic_function

    @symbolic_function.setter
    def symbolic_function(self, function):
        """
        Checks if entered function can be transformed to sympy formatted function.
        """

        if isinstance(function, sp.Expr):
            variables = [(x, sp.Symbol(str(x), real=True)) for x in function.free_symbols]
            function = function.subs(variables)
        elif isinstance(function, str):
            try:
                function = sp.sympify(function)
                variables = [(x, sp.Symbol(str(x), real=True)) for x in function.free_symbols]
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
        if isinstance(function, sp.Expr) or isinstance(function, str):
            variables = self.symbolic_function.free_symbols - {self._symbol}
            variables = sorted(list(variables), key=str)
            variables.insert(0, self._symbol)
            # Create 'fixed' sympy function
            temp_function = symbolic_to_numeric(self.symbolic_function)
            function = sp.lambdify(variables, temp_function, "numpy")
        elif isinstance(function, types.FunctionType) or isinstance(function, sc.interpolate.interp1d):
            function = function
        else:
            raise errors.WrongTypeError(
                "Function must be string/sympy expression/scipy interp1d or FunctionType.")
        self.__numeric_function = function

    def symbolic_derivative(self):
        if self.symbolic_function is not None:
            return sp.diff(self.symbolic_function, self._symbol)
        else:
            raise errors.FunctionNotDefinedError(
                "Symbolic function for '{}' is not defined.".format(self.name))

    def symbolic_second_derivative(self):
        if self.symbolic_function is not None:
            return sp.diff(self.symbolic_function, self._symbol, 2)
        else:
            raise errors.FunctionNotDefinedError(
                "Symbolic function for '{}' is not defined.".format(self.name))

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
            # Take symbolic derivative.
            # Then convert all function symbol strings to sp.Symbol() and convert sympy function to FunctionType.
            symbolic_derivative = self.symbolic_derivative()
            variables = self._parameter_symbols()
            variables = [sp.Symbol(x, real=True) for x in variables]
            variables.insert(0, self._symbol)
            # Create 'fixed' sympy function
            temp_function = symbolic_to_numeric(symbolic_derivative)
            function = sp.lambdify(variables, temp_function, "numpy")
            return function(x, **kwargs)
        else:
            try:
                def function(y):
                    return self.numeric_function(x=y, **kwargs)
            except ZeroDivisionError:
                ZeroDivisionError("Division by zero occured.")
            except:
                raise errors.ParameterDefinitionError(
                    "Inserted number of parameters and function parameters is not same.")
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
            # Take symbolic second derivative.
            # Then convert all function symbol strings to sp.Symbol() and convert sympy function to FunctionType.
            symbolic_derivative = self.symbolic_second_derivative()
            variables = self._parameter_symbols()
            variables = [sp.Symbol(x, real=True) for x in variables]
            variables.insert(0, self._symbol)
            # Create 'fixed' sympy function
            temp_function = symbolic_to_numeric(symbolic_derivative)
            function = np.vectorize(sp.lambdify(variables, temp_function, "numpy"))
            return function(x, **kwargs)
        else:
            try:
                def function(y):
                    return self.numeric_function(x=y, **kwargs)
            except ZeroDivisionError:
                ZeroDivisionError("Division by zero occured.")
            except:
                raise errors.ParameterDefinitionError(
                    "Inserted number of parameters and function parameters is not same.")
            return derivative(func=function, x0=x, n=2, dx=dx)

    def symbolic_function_defined(self):
        return isinstance(self.symbolic_function, sp.Expr)

    def _parameter_symbols(self):
        """
        Function to return all function free parameters.

        Returns
        -------
        tuple
            Function all free parameters' symbol strings.
        """
        return tuple(inspect.getfullargspec(self.numeric_function).args[1:], )


class InflationFunction(InflationFunctionBase):
    """
    More simple class which takes all properties of InflationFunctionBase.
    This function has defined name and settings. Settings are used in calculation.
    Also it has shortened names for easier use later.
    """

    def __init__(self, function, name, settings, **kwargs):
        super().__init__(function, **kwargs)
        self.name = name
        self.settings = settings

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        """
        Define function name.

        Parameters
        ----------
        name : str
            Function new name. Can't be empty ('')
        """
        if isinstance(name, str):
            if name == "":
                raise errors.WrongValueError(
                    "Name can't be empty (name != '').")
            self.__name = name
        else:
            errors.WrongTypeError("Function name must be string.")

    @property
    def settings(self):
        return self.__settings

    @settings.setter
    def settings(self, settings):
        if isinstance(settings, config.Settings):
            self.__settings = settings
        else:
            raise errors.WrongTypeError("Settings must be Settings type.")

    def f_s(self):
        """
        Returns symbolic function
        """
        return self.symbolic_function

    def f_n(self, x, **kwargs):
        """
        Returns numeric function values
        """
        # Take only parameters which are used.
        if self.symbolic_function is not None:
            parameters_dict = {x: kwargs[x] for x in self._parameter_symbols() if x in kwargs}
        else:
            parameters_dict = kwargs
        return self.numeric_function(x, **parameters_dict)

    def d_s(self):
        """
        Return symbolic function derivative.
        """
        return self.symbolic_derivative()

    def dd_s(self):
        """
        Return symbolic function second derivative.
        """
        return self.symbolic_second_derivative()

    def d_n(self, x, **kwargs):
        """
        Return function first derivative values at point x.
        """
        parameters_dict = {x: kwargs[x] for x in self._parameter_symbols()}

        return self.numeric_derivative(x, dx=self.settings.derivative_dx, **parameters_dict)

    def dd_n(self, x, **kwargs):
        """
        Return function second derivative values at point x.
        """
        parameters_dict = {x: kwargs[x] for x in self._parameter_symbols()}

        return self.numeric_second_derivative(x, dx=self.settings.derivative_dx, **parameters_dict)


# Define different function classes
# Used for checking later right types.

class FunctionA(InflationFunction):
    def __init__(self, function, settings, name="A", **kwargs):
        super().__init__(function, name, settings, **kwargs)


class FunctionB(InflationFunction):
    def __init__(self, function, settings, name="B", **kwargs):
        super().__init__(function, name, settings, **kwargs)


class FunctionV(InflationFunction):
    def __init__(self, function, settings, name="V", **kwargs):
        super().__init__(function, name, settings, **kwargs)


class FunctionIV(InflationFunction):
    """
    Function for invariant potential which has varaiable invariant scalar field. I_V (I_φ).
    """

    def __init__(self, function, settings, name="IV", **kwargs):
        super().__init__(function, name, settings, **kwargs)


class FunctionIVf(InflationFunction):
    """
    Function for invariant potential which has varaiable scalar field. I_V (φ).
    """

    def __init__(self, function, settings, name="IVf", **kwargs):
        super().__init__(function, name, settings, **kwargs)


class FunctionIF(InflationFunction):
    """
    Function for invariant scalar field which has varaiable invariant potential. I_φ (I_V).
    """

    def __init__(self, function, settings, name="IF", **kwargs):
        super().__init__(function, name, settings, **kwargs)


class FunctionIFf(InflationFunction):
    """
    Function for invariant scalar field which has variable scalar field. I_φ (φ).
    """

    def __init__(self, function, settings, name="IFf", **kwargs):
        super().__init__(function, name, settings, **kwargs)
