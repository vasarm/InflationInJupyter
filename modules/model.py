import sympy as sp
import numpy as np
from scipy.misc import derivative
from sympy.calculus.util import continuous_domain, minimum
import matplotlib.pyplot as plt

import re
import itertools
import time
import sys

import modules.symbolicfunctions as symbolic_solver
import modules.numericfunctions as numeric_solver

from modules.config import Settings
import modules.plotter as plotter
from modules.function import FunctionA, FunctionB, FunctionV, FunctionIV, symbolic_to_numeric
import modules.errors as errors


def _create_parameter_key(parameter_combination):
    """
    Create key value for parameter combination.

    Parameters
    ----------
    parameter_combination : dict
        One parameter combination used for calculation.

    Returns
    -------
    str
        Key value for that combination
    """
    temp_list = []
    for key, value in sorted(parameter_combination.items(), key=lambda x: str(x[0])):
        temp_list.append(str(key) + "=" + str(value))
    return ", ".join(temp_list)


def _convert_dictionary_symbols_to_strings(dictionary):
    return dict((str(key), dictionary[key]) for key in dictionary)


def _ipython_info():
    """
    To let program know if this is runned by jupyter notebook/lab or not.

    Returns
    -------
    string
        Returns what module is used.
    """
    ip = False
    if 'ipykernel' in sys.modules:
        ip = True
    elif 'IPython' in sys.modules:
        ip = 'terminal'
    return ip


def _convert_dictionary_strings_to_symbols(dictionary):
    """
    Used to convert parameter combination which keys are strings to keys which are sp.Symbol type.

    Parameters
    ----------
    dictionary : Parameter combination
        Parameter combination
    """
    return dict((sp.Symbol(str(key), real=True), dictionary[key]) for key in dictionary)


class BaseModel:

    def __init__(self, settings, A=None, B=None, V=None, IV=None, formalism="Metric", **kwargs):
        """
        Base class for Inflation model. This defines single combination of (A,B,V) or (IV) as model
        which is later used in other class to calculate observable parameters.

        Parameters
        ----------
        settings : Settings
            Program settings
        A : FunctionA, optional
            Required if calculation is done with A, B and V functions, by default None
        B : FunctionB, optional
            Required if calculation is done with A, B and V functions, by default None
        V : FunctionV, optional
            Required if calculation is done with A, B and V functions, by default None
        IV : FunctionIV, optional
            Required if calculation is done with only IV, by default None
        """
        self.settings = settings
        self.settings.formalism = formalism
        self.mode = None

        self.A = A
        self.B = B
        self.V = V
        self.IV = IV
        # Three possible variants:
        # 0 = Mode not defined
        # 1 = Function mode
        # 2 = Invariant mode
        self.update_mode()
        self._symbol = sp.Symbol("x", real=True)

        #
        # "Write condition that A, B and V or IV are defined"
        if all(x is not None for x in [self.A, self.B, self.V]) or self.IV is not None:
            self.parameter_values = self._define_parameter_values()
            self.parameter_combinations = self._combine_parameter_values()

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        if isinstance(mode, int) and mode in [0, 1, 2, 3]:
            self.__mode = mode
        else:
            self.__mode = 0
            # print("For calculations define (A,B,V) functions or IV function.")

    def update_mode(self):
        if all([x is not None for x in [self.A, self.B, self.V]]) and self.IV is not None:
            self.mode = 3
        elif all([x is not None for x in [self.A, self.B, self.V]]):
            self.mode = 1
        elif self.IV is not None:
            self.mode = 2
        else:
            self.mode = 0

    @property
    def A(self):
        return self.__A

    @A.setter
    def A(self, A):
        if isinstance(A, FunctionA):
            self.__A = A
        elif isinstance(A, str):
            self.__A = FunctionA(A, self.settings)
        elif A is None:
            self.__A = None
        else:
            raise errors.WrongTypeError("Function A must be FunctionA type.")

        if all(hasattr(self, x) for x in ["_BaseModel__A", '_BaseModel__B', '_BaseModel__V', '_BaseModel__IV']):
            self.update_mode()

    @property
    def B(self):
        return self.__B

    @B.setter
    def B(self, B):
        if isinstance(B, FunctionB):
            self.__B = B
        elif isinstance(B, str):
            self.__B = FunctionB(B, self.settings)
        elif B is None:
            self.__B = None
        else:
            raise errors.WrongTypeError("Function B must be FunctionB type.")
        if all(hasattr(self, x) for x in ["_BaseModel__A", '_BaseModel__B', '_BaseModel__V', '_BaseModel__IV']):
            self.update_mode()

    @property
    def V(self):
        return self.__V

    @V.setter
    def V(self, V):
        if isinstance(V, FunctionV):
            self.__V = V
        elif isinstance(V, str):
            self.__V = FunctionV(V, self.settings)
        elif V is None:
            self.__V = None
        else:
            raise errors.WrongTypeError("Function V must be FunctionV type.")

        if all(hasattr(self, x) for x in ["_BaseModel__A", '_BaseModel__B', '_BaseModel__V', '_BaseModel__IV']):
            self.update_mode()

    @property
    def IV(self):
        return self.__IV

    @IV.setter
    def IV(self, IV):
        if isinstance(IV, FunctionIV):
            self.__IV = IV
        elif isinstance(IV, str):
            self.__IV = FunctionIV(IV, self.settings)
        elif IV is None:
            self.__IV = None
        else:
            raise errors.WrongTypeError("Function IV must be FunctionIV type.")

        if all(hasattr(self, x) for x in ["_BaseModel__A", '_BaseModel__B', '_BaseModel__V', '_BaseModel__IV']):
            self.update_mode()

    def _all_functions_symbolic(self):
        """
        Check if all functions have symbolic function defined.

        Returns
        -------
        bool
            True if all functions have symbolic funtion. Else False.

        Raises
        ------
        errors.ModeError
            If self.mode == 0 in other words mode is not defined because combination of (A,B,V) or IV is not defined.
        """
        if self.mode == 1 or self.mode == 3:
            return all(x.symbolic_function_defined() for x in [self.A, self.B, self.V])
        elif self.mode == 2:
            return self.IV.symbolic_function_defined()
        else:
            raise errors.ModeError("Can't check if all functions are symbolic\n" +
                                   "because combination of (A,B,V) functions or IV function is not defined.")

    def _return_parameter_symbols(self):
        """
        Returns function free parameters which are tuple of strings.

        Returns
        -------
        tuple
            Function free parameters.

        Raises
        ------
        errors.ModeError
            If self.mode == 0 in other words mode is not defined because combination of (A,B,V) or IV is not defined.
        """
        if self.mode == 1 or self.mode == 3:
            symbols = list(set(self.A._parameter_symbols(
            ) + self.B._parameter_symbols() + self.V._parameter_symbols()))
            symbols.sort()

        elif self.mode == 2:
            symbols = self.IV._parameter_symbols()
        else:
            raise errors.ModeError("Can't get function free parameters\n" +
                                   "because combination of (A,B,V) functions or IV function is not defined.")
        return tuple(symbols)

    def _return_parameter_symbolic_symbols(self):
        """
        Returns function free parameter symbols which types are sp.Symbol.

        Returns
        -------
        tuple
            Function free parameters.

        Raises
        ------
        errors.ModeError
            If self.mode == 0 in other words mode is not defined because combination of (A,B,V) or IV is not defined.
        """
        if self.mode == 1 or self.mode == 3:
            symbols = self.A.f_s().free_symbols | self.B.f_s().free_symbols | self.V.f_s().free_symbols
            symbols = symbols - {self._symbol}
            symbols = list(symbols)
            symbols.sort(key=str)
        elif self.mode == 2:
            symbols = sorted(
                list(self.IV.f_s().free_symbols - {self._symbol}), key=str)
        else:
            raise errors.ModeError("Can't get function free parameters\n" +
                                   "because combination of (A,B,V) functions or IV function is not defined.")
        return tuple(symbols)

    def _define_parameter_values(self):
        """
        If model is initiated then program asks values for each free parameter.

        Returns
        -------
        dict
            Returns dictionary where keys are parameter symbol (as strings) and key values are user defined numerical
             values.

        Raises
        ------
        errors.ModeError
            self.mode = 0 error. Can't proceed with calculation because not enough functions defined.
        """
        # To split user numerical input
        if self.mode == 0:
            raise errors.ModeError(
                "Can't define parameter values. User must define combination of (A,B,V) functions or IV function to "
                "carry on.")

        parameter_value_dict = {}
        decimal_regex = re.compile("[-+]?\d*\.\d+|[-+]?\d+")

        # Sorted tuple of parameter symbols
        parameters = self._return_parameter_symbols()

        print("Kõik sümbolid on: {}".format(parameters))
        for parameter in parameters:
            while True:
                try:
                    print(
                        "Sisesta parameetri '{}' väärtused. Komaga arvude jaoks kasuta ('.'). \n Numbrid eralda komadega või tühikuga.".format(
                            parameter))
                    user_input = input(
                        "Siseta '{}' väärtus(ed) siia --->: ".format(parameter))
                    # Find all possible parameter values.
                    user_input = np.unique(
                        [np.int(x) if float(x).is_integer() else np.float(x) for x in
                         decimal_regex.findall(user_input)])
                    if len(user_input) == 0:
                        raise errors.WrongValueError(
                            "Kasutaja peab sisestama vähemalt ühe väärtuse.")
                    break
                except:
                    print("Proovi uuesti. Probleem sisestamisel või konverteerimisel.")
            parameter_value_dict[str(parameter)] = user_input

        return parameter_value_dict

    def _combine_parameter_values(self):
        """
        Creates all possible combinations of parameters.

        Returns
        -------
        tuple
            Returns tuple of all combinations. One combination is a dictionary where key is parameter symbol
            and key value is one possible parameter value.
            For example:
            symbols are (a, b, c) and each has 3 values then there are 27 combinations. Each combination is a dictionary
            dict = {a: num, b:num, c:num}.
        """
        parameter_symbols = self._return_parameter_symbols()
        number_of_symbols = len(parameter_symbols)
        all_parameter_values = [self.parameter_values[x]
                                for x in parameter_symbols]

        combined_values = tuple(itertools.product(*all_parameter_values))
        combinations = tuple({parameter_symbols[i]: combination[i] for i in range(
            number_of_symbols)} for combination in combined_values)

        return combinations

    def return_formalism(self):
        return self.settings.formalism

    def change_inteval(self):
        print("Muuda skalaarvälja määramispiirkonda.")
        print("Hetkel on: [{}, {} ]".format(self.settings.scalar_field_range[0], self.settings.scalar_field_range[1]))
        def ask_start():
            try:
                start_value = np.float(input("Sisesta algväärtus: "))
                return start_value
            except:
                print("Error. Viga teisendamisel.")
                ask_start()

        start = ask_start()

        def ask_end():
            try:
                end_value = np.float(input("Sisesta lõppväärtus: "))
                if end_value >= start:
                    raise TypeError
                return end_value
            except TypeError:
                print("Error. Lõppväärtus peab olema väiksem kui algväärtus.")
                ask_end()
            except:
                print("Error. Viga teisendamisel.")
                ask_end()

        end = ask_end()

        self.settings.scalar_field_range = [start, end]

class CalculationModel(BaseModel):
    """
    In this class all possible calculations with inflation functions are defined. Used to calculate F, scalar-tensor
    pertubation ratio r, scalar spectral index ns, epsilon ε which defines inflation end point.
    """

    def __init__(self, settings, A=None, B=None, V=None, IV=None, **kwargs):
        super().__init__(settings, A=A, B=B, V=V, IV=IV, **kwargs)

    def F_s(self):
        """
        Calculate function F which is combination of A, B and V.

        Returns
        -------
        sp.Expr
            F function expression

        Raises
        ------
        errors.FormalismError
            If formalism in settings is wrong. Must be "Metric" or "Palatini".
        errors.AllFunctionsNotSymbolic
            If program tries to calculate symbolically but not all functions are symbolic.
        """
        if self._all_functions_symbolic():
            if self.settings.formalism == "Metric":
                return self.B.f_s() / self.A.f_s() + 3 / 2 * (self.A.d_s() / self.A.f_s()) ** 2
            elif self.settings.formalism == "Palatini":
                return self.B.f_s() / self.A.f_s()
            else:
                raise errors.FormalismError(
                    "Formalism must be 'Metric' or 'Palatini'.")
        else:
            raise errors.AllFunctionsNotSymbolic(
                "All functions are not symbolically defined.")

    def F_n(self, x, **kwargs):
        """
        Calculate function F values which is combination of A, B and V.

        Returns
        -------
        array_like
            F function values in position x

        Raises
        ------
        errors.FormalismError
            If formalism in settings is wrong. Must be "Metric" or "Palatini".
        """
        if self.settings.formalism == "Metric":
            return self.B.f_n(x=x, **kwargs) / (self.A.f_n(x=x, **kwargs) + self.settings.divisor_inc) + 3 / 2 * (
                    self.A.d_n(x=x, **kwargs) / (self.A.f_n(x=x, **kwargs) + self.settings.divisor_inc)) ** 2
        elif self.settings.formalism == "Palatini":
            return self.B.f_n(x=x, **kwargs) / (self.A.f_n(x=x, **kwargs) + self.settings.divisor_inc)
        else:
            raise errors.FormalismError(
                "Formalism must be 'Metric' or 'Palatini'.")

    def dF_s(self):
        """
        Calculate function F derivative.

        Returns
        -------
        sp.Expr
            F function derivative expression

        Raises
        ------
        errors.AllFunctionsNotSymbolic
            If program tries to calculate symbolically but not all functions are symbolic.
        """
        if self._all_functions_symbolic():
            return sp.diff(self.F_s(), self._symbol)
        else:
            raise errors.AllFunctionsNotSymbolic(
                "All functions are not symbolically defined.")

    def dF_n(self, x, **kwargs):
        """
        Calculate function F derivative values. If possible use symbolic functions to define function.

        Returns
        -------
        array_like
            F function derivative values in position x
        """
        if self._all_functions_symbolic() and self.settings.symbolic_to_numeric:
            symbolic_dF = symbolic_to_numeric(self.dF_s())
            symbols = (self._symbol,) + self._return_parameter_symbolic_symbols()
            numeric_dF = sp.lambdify(symbols, symbolic_dF, "numpy")
            return numeric_dF(x, **kwargs)
        else:
            temp_F = lambda y: self.F_n(x=y, **kwargs)
            return derivative(func=temp_F, x0=x, dx=self.settings.derivative_dx, n=1)

    def r_s(self):
        """
        Calulate observable tesnor-to-scalar pertubation ratio value.

        Returns
        -------
        sp.Expr
            Tensor-to-scalar pertubation ratio expression.

        Raises
        ------
        errors.AllFunctionsNotSymbolic
            If program tries to calculate symbolically but not all functions are symbolic.
        """
        # Case where are defined A, B, V
        if self._all_functions_symbolic() and self.mode == 1 or self.mode == 3:
            return (8 / self.F_s()) * (
                    (self.V.d_s() * self.A.f_s() - 2 * self.V.f_s() * self.A.d_s())
                    / (self.A.f_s() * self.V.f_s())
            ) ** 2
        # Case where is defined IV
        elif self._all_functions_symbolic() and self.mode == 2:
            return 16 * self.epsilon_s()
        else:
            raise errors.AllFunctionsNotSymbolic(
                "All functions are not symbolically defined.")

    def r_n(self, x, **kwargs):
        """
        Calculate tensor-to-scalar pertubation ratio values. If possible use symbolic functions to define function
        and then calculate values.

        Returns
        -------
        array_like
            F function derivative values in position x
        """
        if self._all_functions_symbolic() and self.settings.symbolic_to_numeric:
            symbols = (self._symbol,) + \
                      self._return_parameter_symbolic_symbols()
            temp_function = symbolic_to_numeric(self.r_s())
            numeric_r = sp.lambdify(symbols, temp_function, "numpy")
            return numeric_r(x, **kwargs)
        else:
            # A, B and V are defined
            if self.mode == 1 or self.mode == 3:
                return (8 / (self.F_n(x=x, **kwargs) + self.settings.divisor_inc)) * (
                        (self.V.d_n(x=x, **kwargs) * self.A.f_n(x=x, **kwargs) -
                         2 * self.V.f_n(x=x, **kwargs) * self.A.d_n(x=x, **kwargs))
                        / (self.A.f_n(x=x, **kwargs) * self.V.f_n(x=x, **kwargs) + self.settings.divisor_inc)
                ) ** 2
            # IV is defined
            elif self.mode == 2:
                return 16 * self.epsilon_n(x, **kwargs)

    def ns_s(self):
        """
        Calulate observable scalar spectral index pertubation ratio expression.

        Returns
        -------
        sp.Expr
            scalar spectral index expression.

        Raises
        ------
        errors.AllFunctionsNotSymbolic
            If program tries to calculate symbolically but not all functions are symbolic.
        """
        # Functions A, B and V are defined
        if self._all_functions_symbolic() and (self.mode == 1 or self.mode == 3):
            return 1 - 3 * self.r_s() / 8 + (2 / (self.V.f_s() * self.F_s())) * (
                    self.V.dd_s() - 4 * self.V.d_s() * self.A.d_s() / self.A.f_s() -
                    self.V.d_s() * self.dF_s() / (2 * self.F_s()) -
                    2 * self.V.f_s() * self.A.dd_s() / self.A.f_s() +
                    6 * self.V.f_s() * (self.A.d_s() / self.A.f_s()) ** 2 +
                    (self.V.f_s() * self.A.d_s() * self.dF_s()) /
                    (self.A.f_s() * self.F_s())
            )
        # Function IV is defined
        elif self._all_functions_symbolic() and self.mode == 2:
            return 1 - 6 * self.epsilon_s() + 2* self.IV.dd_s() / self.IV.f_s()
        else:
            raise errors.AllFunctionsNotSymbolic(
                "All functions are not symbolically defined.")

    def ns_n(self, x, **kwargs):
        """
        Calculate scalar spectral index values. If possible use symbolic functions to define function and then
        calculate values.

        Returns
        -------
        array_like
            scalar spectral index n_s  values in position x
        """
        if self._all_functions_symbolic() and self.settings.symbolic_to_numeric:
            symbols = (self._symbol,) + \
                      self._return_parameter_symbolic_symbols()
            temp_function = symbolic_to_numeric(self.ns_s())
            numeric_n = sp.lambdify(symbols, temp_function, "numpy")
            return numeric_n(x, **kwargs)
        else:
            # Functions A, B and V are defined
            if self.mode == 1 or self.mode == 3:
                return 1 - 3 * self.r_n(x=x, **kwargs) / 8 + (
                        2 / (self.V.f_n(x=x, **kwargs) * self.F_n(x=x, **kwargs) + self.settings.divisor_inc)) * (
                               self.V.dd_n(x=x, **kwargs) -
                               4 * self.V.d_n(x=x, **kwargs) * self.A.d_n(x=x, **kwargs) / (
                                       self.A.f_n(x=x, **kwargs) + self.settings.divisor_inc) -
                               self.V.d_n(x=x, **kwargs) * self.dF_n(x=x, **kwargs) / (
                                       2 * self.F_n(x=x, **kwargs) + self.settings.divisor_inc) -
                               2 * self.V.f_n(x=x, **kwargs) * self.A.dd_n(x=x, **kwargs) / (
                                       self.A.f_n(x=x, **kwargs) + self.settings.divisor_inc) +
                               6 * self.V.f_n(x=x, **kwargs) * (
                                       self.A.d_n(x=x, **kwargs) / (
                                       self.A.f_n(x=x, **kwargs) + self.settings.divisor_inc)) ** 2 +
                               (self.V.f_n(x=x, **kwargs) * self.A.d_n(x=x, **kwargs) * self.dF_n(x=x, **kwargs)) /
                               (self.A.f_n(x=x, **kwargs) * self.F_n(x=x, **kwargs) + self.settings.divisor_inc)
                       )
            # Function IV is defined
            elif self.mode == 2:
                return 1 - 6 * self.epsilon_n(x, **kwargs) + 2 * self.IV.dd_n(x, **kwargs) * (self.IV.f_n(x=x, **kwargs) + self.settings.divisor_inc)

    def N_integrand_s(self):
        """
        Calulate N_integrand expression. This is defined to integrate and find function N(φ).

        Returns
        -------
        sp.Expr
            N_integrand expression.

        Raises
        ------
        errors.AllFunctionsNotSymbolic
            If program tries to calculate symbolically but not all functions are symbolic.
        """
        # Functions A, B and V are defined
        if self._all_functions_symbolic() and (self.mode == 1 or self.mode == 3):
            return ((self.A.f_s() * self.V.f_s() * self.F_s()) /
                    (self.V.d_s() * self.A.f_s() - 2 * self.V.f_s() * self.A.d_s())
                    )
        # Function IV is defined
        elif self._all_functions_symbolic() and self.mode == 2:
            return self.IV.f_s() / self.IV.d_s()
        else:
            raise errors.AllFunctionsNotSymbolic(
                "All functions are not symbolically defined.")

    def N_integrand_n(self, x, **kwargs):
        """
        Calculate N_integrand values. If possible use symbolic functions to define function and then calculate values.
        This is used to numerically integrate scalar field values to find N values.

        Returns
        -------
        array_like
            N_integrand values in position x
        """
        if self._all_functions_symbolic() and self.settings.symbolic_to_numeric:
            symbols = (self._symbol,) + \
                      self._return_parameter_symbolic_symbols()
            temp_function = symbolic_to_numeric(self.N_integrand_s())
            numeric_N = sp.lambdify(symbols, temp_function, "numpy")
            return numeric_N(x, **kwargs)
        else:
            # Functions A, B and V are defined
            if self.mode == 1 or self.mode == 3:
                return ((self.A.f_n(x=x, **kwargs) * self.V.f_n(x=x, **kwargs) * self.F_n(x=x, **kwargs)) /
                        (self.V.d_n(x=x, **kwargs) * self.A.f_n(x=x, **kwargs) -
                         2 * self.V.f_n(x=x, **kwargs) * self.A.d_n(x=x, **kwargs) + self.settings.divisor_inc)
                        )
            # Function IV is defined
            elif self.mode == 2:
                return self.IV.f_n(x, **kwargs) / (self.IV.d_n(x, **kwargs) + self.settings.divisor_inc)

    def epsilon_s(self):
        """
        Calulate epsilon (ε) expression.
        This is used to find scalar field value where inflation ends. This means that ε(φ) = 1.

        Returns
        -------
        sp.Expr
            epsilon (ε) expression.

        Raises
        ------
        errors.AllFunctionsNotSymbolic
            If program tries to calculate symbolically but not all functions are symbolic.
        """
        # Functions A, B and V are defined
        if self._all_functions_symbolic() and (self.mode == 1 or self.mode == 3):
            return self.r_s() / 16
        # Function IV is defined
        if self._all_functions_symbolic() and self.mode == 2:
            return 1 / 2 * (self.IV.d_s() / self.IV.f_s()) ** 2
        else:
            raise errors.AllFunctionsNotSymbolic(
                "All functions are not symbolically defined.")

    def epsilon_n(self, x, **kwargs):
        """
        Calculate epsilon (ε) values. If possible use symbolic functions to define function and then calculate values.
        This is used to find scalar field value where inflation ends. This means that ε(φ) = 1.

        Returns
        -------
        array_like
            epsilon (ε) values in position x
        """
        if self._all_functions_symbolic() and self.settings.symbolic_to_numeric:
            symbols = (self._symbol,) + \
                      self._return_parameter_symbolic_symbols()
            temp_function = symbolic_to_numeric(self.epsilon_s())
            return sp.lambdify(symbols, temp_function, "numpy")(x, **kwargs)
        else:
            # Functions A, B and V are defined
            if self.mode == 1 or self.mode == 3:
                return self.r_n(x, **kwargs) / 16
            # Function IV is defined
            elif self.mode == 2:
                return 1 / 2 * (self.IV.d_n(x, **kwargs) / (self.IV.f_n(x, **kwargs) + self.settings.divisor_inc)) ** 2


class InflationModel(CalculationModel):
    """
    Main class, which end-user uses. This class has functions which define calculation order and calculated.
    """

    def __init__(self, settings=None, A=None, B=None, V=None, IV=None, name=None, **kwargs):
        if settings is None:
            settings = Settings()

        super().__init__(settings, A=A, B=B, V=V, IV=IV, **kwargs)

        # Dictionaries which keys are defined by parameters and their values. Uses function _create_parameter_key
        self.end_values = {}
        self.N_functions = {}
        self.name = name

    def _ask_function(self, function_name):
        """
        Function which asks user to define function expression.

        Parameters
        ----------
        function_name : string
            Function name
        """
        print("Sisesta funktsioon {}.".format(function_name))
        while True:
            user_input = input("Siia--> ")
            if function_name == "A":
                try:
                    self.A = FunctionA(user_input, self.settings)
                except:
                    print("Viga sisestusel. Proovi uuesti.")
            elif function_name == "B":
                try:
                    self.B = FunctionB(user_input, self.settings)
                except:
                    print("Viga sisestusel. Proovi uuesti.")
            elif function_name == "V":
                try:
                    self.V = FunctionV(user_input, self.settings)
                except:
                    print("Viga sisestusel. Proovi uuesti.")
            if function_name == "IV":
                try:
                    self.IV = FunctionIV(user_input, self.settings)
                except:
                    print("Viga sisestusel. Proovi uuesti.")

    def initialize(self):
        """
        Program to initialize model. Asks user to insert required functions. This asks user to enter
        what functions to use for calculation: 1) A, B and V or 2) only IV
        """
        print("Choose what functions to use:\n1) A, B, V\n2) I_V")
        while True:
            user_input = input("Enter mode number --> ")
            if user_input in ["1", "2"]:
                mode = int(user_input)
                break
            else:
                print("Enter correct value.")

        if mode == 1:
            print("olen siin")
            self._ask_function("A")
            self._ask_function("B")
            self._ask_function("V")
            print("Entered functions are:")
            print("A:", self.A.f_s())
            print("B:", self.B.f_s())
            print("V:", self.V.f_s())
        elif mode == 2:
            self._ask_function("IV")
            print("Entered function is:")
            print("IV:", self.IV)
        self.mode = int(mode)

        self.parameter_values = self._define_parameter_values()
        self.parameter_combinations = self._combine_parameter_values()

    def domain_is_continuous(self, parameter_combination, *functions):
        try:
            parameter_values = _convert_dictionary_strings_to_symbols(parameter_combination)
            settings_domain = sp.Interval(*self.settings.scalar_field_range)
            functions_domain = sp.Reals
            for func in functions:
                func_domain = continuous_domain(func.subs(parameter_values), self._symbol, sp.Reals)
                functions_domain = functions_domain.intersect(func_domain)

            if not settings_domain.intersect(functions_domain) == settings_domain:
                def convert_interval(interval):
                    interval = str(interval)
                    if interval == "Reals":
                        return "Reals"
                    answer = []
                    interval_list = interval.split("Interval")
                    for sub in interval_list:
                        search = re.search("\(.*\)", str(sub))
                        if search:
                            result = search.group(0).replace("))", ")")
                            if "open" not in sub:
                                result = result.replace("(", "[").replace(")", "]").replace("oo]", "oo)").replace(
                                    "[-oo", "(-oo")
                            answer.append(result)
                    return " U ".join(answer)

                print("\n***** Hoiatus! Tulemused võivad olla mitte usaldusväärsed. ******")
                print("Probleem skalaarvälja φ määramispiirkonnaga.")
                print("Sätete φ määramispiirkond ∈ {}".format(convert_interval(settings_domain)))
                print("Funktsioonide φ määramispiirkond  ∈ {}".format(convert_interval(functions_domain)))
                print("***** Hoiatus! Tulemused võivad olla mitte usaldusväärsed. ******\n")
            else:
                print("***** Skalaarvälja määramispiirkond peaks olema OK. *****")
        except:
            print("***** Ei saanud kontrollida skalaarvälja määramispiirkonna sobivust. *****")

    def domain_is_valid(self, parameter_combination):
        # Check if A and B/A + 3/2(A'/A)^2 is valid - in other wordse these functions must be positive.
        try:
            parameter_values = _convert_dictionary_strings_to_symbols(parameter_combination)
            domain_settings = sp.Interval(*self.settings.scalar_field_range)
            a_minimum = minimum(self.A.f_s().subs(parameter_values), self._symbol, domain_settings)
            ba_minimum = minimum(self.F_s().subs(parameter_values), self._symbol, domain_settings)
            if a_minimum > 0 and ba_minimum > 0:
                print("***** A(φ) ja F(φ) funktsioonid määratud piirkonnas on positiivsed. *****")
            else:
                print("***** A(φ) ja F(φ) funktsioonid määratud piirkonnas ei ole alati positiivsed! *****")
                print("***** Programmi tulemused võivad olla valed *****")
        except:
            print("***** Ei saanud kontrollida A(φ) ja F(φ) funktsioonide sobivust. *****")

    def check_models(self, **kwargs):
        def compare_results(res1, res2):
            return np.abs((res1 - res2) / res2)

        if not all(x.symbolic_function_defined() for x in [self.A, self.B, self.V, self.IV]):
            print("KõiK funktsioonid A, B, V ja IV peavad olema defineeritud.")
            return

        symbols = self.A.f_s().free_symbols | self.B.f_s().free_symbols | self.V.f_s().free_symbols | self.IV.f_s().free_symbols
        symbols = symbols - {self._symbol}
        symbols = set([str(symbol) for symbol in symbols])
        inserted_symbols = set(kwargs.keys())
        if not inserted_symbols == symbols:
            print("Kõik parameetrid ei ole defineeritud.")
            print("Defineeri parameetrid: ", inserted_symbols ^ symbols)
        print("1) Otsin pöördfunktsioone.")
        try:
            parameter_values = _convert_dictionary_strings_to_symbols(dict(kwargs))
            IF_solutions = symbolic_solver.run_IF_calculation_symbolical(self.IV.f_s().subs(parameter_values),
                                                                         self.settings.timeout)
            VA2 = self.V.f_s().subs(parameter_values) / self.A.f_s().subs(parameter_values) ** 2
            # Find all invariant scalar field derivatives
            dIF_functions = [sp.diff(func.subs(self._symbol, VA2), self._symbol) for func in IF_solutions]
            dIF2_num = [sp.lambdify(self._symbol, x, "numpy") for x in dIF_functions]

        except Exception as e:
            print(e)
            print("Ei saanud arvutada pöördfunktsiooni sümbolkujul.")
            print("Ei saa kontrollida mudelite samasust.")
            return
        print("   Pöördfunktsioon(id) leitud.")
        domain = self.settings.scalar_field_domain_plot
        print("2) Arvutan funktsioonide väärtused.")
        with np.errstate(divide="ignore", invalid="ignore"):
            A_values = self.A.f_n(domain, **kwargs)
            F_values = self.F_n(domain, **kwargs)
            if isinstance(A_values, int) or isinstance(A_values, float):
                A_values = np.full(domain.shape[0], A_values)
            if isinstance(F_values, int) or isinstance(F_values, float):
                F_values = np.full(domain.shape[0], F_values)

            A_pos = (A_values > 0) & (A_values != np.inf) & (A_values != np.nan) & (A_values != -np.inf)
            F_pos = (F_values > 0) & (F_values != np.inf) & (F_values != np.nan) & (F_values != -np.inf)

            AF_pos = A_pos & F_pos
            valid_domain = domain[AF_pos]

            # Calculate derivative values
            dIF1 = np.sqrt(self.F_n(valid_domain, **kwargs))
            if isinstance(dIF1, int) or isinstance(dIF1, float):
                dIF1 = np.full(valid_domain.shape[0], dIF1)

            dIF2 = [func(valid_domain) for func in dIF2_num]
            dIF2 = [np.full(valid_domain.shape[0], func) if isinstance(func, int) or isinstance(func, float) else func for func in dIF2]

            comparisons = [compare_results(dIF1, array) for array in dIF2]
            comparisons = [np.full(valid_domain.shape[0], comp) if isinstance(comp, int) or isinstance(comp, float) else comp for comp in
                           comparisons]

            names = [sp.latex(func) for func in IF_solutions]
            BAname = sp.latex(sp.sqrt(self.F_s().subs(parameter_values)))

            print("3) Koostan graafikuid.")
            plotter.compare_models(valid_domain, dIF1, dIF2, comparisons, BAname, names)

    def _find_field_end_value(self, parameter_combination, method="numeric", return_value=False, info=True,
                              subprocess_info=False):
        """
        Define process to calculate scalar field value in the end of inflation.

        Parameters
        ----------
        parameter_combination : dict
            Combination of parameters
        method : str, optional
            symbolic or numeric, by default "numeric"
        return_value : bool, optional
            Does function returns calculated value?, by default False
        info : bool, optional
            Does function prints information about calculation?, by default True

        Returns
        -------
        bool
            if return_value = True then returns scalar field end value

        Raises
        ------
        errors.WrongValueError
            In case method != "symbolic" or "numeric".
        """
        # First define used key
        key = _create_parameter_key(parameter_combination)

        # Check defined scalar field domain compatibility
        if self.mode in [1, 3]:
            self.domain_is_continuous(parameter_combination, self.A.f_s(), self.B.f_s(), self.V.f_s(), self.epsilon_s())
        else:
            self.domain_is_continuous(parameter_combination, self.IV.f_s(), self.epsilon_s())

        with np.errstate(divide="ignore", invalid="ignore"):
            if info:
                print("***** Arvutan skalaarvälja lõppväärtust *****")
                print("***** {} *****".format(key))
                start_time = time.perf_counter()
            if method == "symbolic":
                try:
                    if info:
                        print("***** Arvutan sümboolselt *****")
                    value = self._find_field_end_value_symbolic(
                        parameter_combination, info, subprocess_info)
                except (errors.TimeoutError, errors.NoSolutionError) as e:
                    print(e)
                    print(
                        "Ei saanud sümboolselt arvutada. Proovin lahendada numbriliselt.")
                    value = self._find_field_end_value_numeric(
                        parameter_combination, info, subprocess_info)
            elif method == "numeric":
                if info:
                    print("***** Arvutan numbriliselt *****")
                value = self._find_field_end_value_numeric(
                    parameter_combination, info, subprocess_info)
            else:
                raise errors.WrongValueError(
                    "method peab olema 'symbolic' või 'numeric'.")

            if info:
                end_time = time.perf_counter()
                print("***** Inflatsioon lõppeb φ = {:.6f} *****".format(value))
                print("Aeg: {:.2f} s.".format(end_time - start_time))
                print("***** Skalaarvälja lõppväärtus leitud. *****")

            self.end_values[key] = value

            if return_value:
                return value

    def _find_field_end_value_symbolic(self, parameter_combination, info, subprocess_info):
        """
        Process runs function to calculate scalar field end value symbolically.
        After solution has been found it sorts possible solutions and asks user input if needed.

        Parameters
        ----------
        parameter_combination : dictionary
            Parameter combination
        info : bool
            Does it print calculation info
        subprocess_info : bool
            Does it print function expression

        Returns
        -------
        float
            Scalar field end value

        Raises
        ------
        errors.NoSolutionError
            Symbolically couldn't find any solution for scalar field end value.
        """
        parameter_combination_sym = _convert_dictionary_strings_to_symbols(
            parameter_combination)

        # Substitute parameter values in epsilon function
        epsilon = self.epsilon_s().subs(parameter_combination_sym)

        # Usually should be turned off as it slows down and no effect.
        if self.settings.simplify:
            print("Simplifying:")
            epsilon = sp.simplify(
                epsilon, inverse=self.settings.simplify_inverse, rational=self.settings.simplify_rational)

        if subprocess_info:
            print("ε(φ)-1 funktsioon : {}=0".format(sp.latex(epsilon - 1)))

        if info:
            print("***** Otsin nullkohti *****")
            start_time = time.perf_counter()

        end_value_list = symbolic_solver.run_end_value_calculation_symbolical(
            epsilon, time=self.settings.timeout)

        if info:
            end_time = time.perf_counter()
            print("***** Nullkohad leitud *****")
            print("Aeg: {:.2f} s.".format(end_time - start_time))
        # Checks if end values are in predefined interval
        end_value_list = np.array([x for x in end_value_list if
                                   self.settings.scalar_field_range[0] <= x <= self.settings.scalar_field_range[1]])

        if len(end_value_list) == 0:
            raise errors.NoSolutionError(
                "Nullkohti ei leitud. Proovi muuta määramispiirkonda.")
        elif len(end_value_list) == 1:
            end_value = end_value_list[0]
        else:
            # Create used function list
            plotter_functions = {"V": self.V.f_n, "e": self.epsilon_n}

            figure = plotter.plot_scalar_field_end_values(
                plotter_functions, parameter_combination, end_value_list, self.settings.scalar_field_domain_plot)
            # Display picture with plt.show() in jupyter, as only this is viable method.
            if not _ipython_info():
                figure.show()
            else:
                plt.show()
            end_value = plotter.ask_scalar_field_end_value(end_value_list)

            if _ipython_info():
                plt.close()

        return np.float(end_value)

    def _find_field_end_value_numeric(self, parameter_combination, info, subprocess_info=False):
        """
        Process runs function to calculate scalar field end value numerically.
        After solution has been found it sorts possible solutions and asks user input if needed.

        Parameters
        ----------
        parameter_combination : dictionary
            Parameter combination
        info : bool
            Does it print calculation info
        subprocess_info : bool
            Not needed.

        Returns
        -------
        float
            Scalar field end value

        Raises
        ------
        errors.NoSolutionError
            Numerically couldn't find any solution for scalar field end value.
        """
        if self._all_functions_symbolic():
            parameter_combination_sym = _convert_dictionary_strings_to_symbols(
                parameter_combination)
            epsilon = self.epsilon_s().subs(parameter_combination_sym)
            if self.settings.simplify:
                print("***** Simplifying *****")
                epsilon = sp.simplify(
                    epsilon, inverse=self.settings.inverse, rational=self.settings.rational)
                print("***** Simplified *****")

            func_derivative = sp.diff(epsilon - 1, self._symbol)
            func_derivative2 = sp.diff(epsilon - 1, self._symbol, 2)
            epsilon_minus_one = sp.lambdify(self._symbol, epsilon - 1, "numpy")
            fprime = sp.lambdify(self._symbol, func_derivative, "numpy")
            fprime2 = sp.lambdify(self._symbol, func_derivative2, "numpy")
        else:
            def epsilon_minus_one(x):
                return self.epsilon_n(
                    x, **parameter_combination) - 1

            def fprime(x):
                return derivative(
                    func=epsilon_minus_one, x0=x, dx=self.settings.derivative_dx)

            def fprime2(x):
                return derivative(
                    func=epsilon_minus_one, x0=x, dx=self.settings.derivative_dx, n=2)

        # Lets find approximate zeros
        end_value_list = numeric_solver.find_function_roots_numerical(
            epsilon_minus_one, self.settings)

        if len(end_value_list) == 0:
            raise errors.NoSolutionError(
                "No end value found. Changing interval might help.")
        elif len(end_value_list) == 1:
            selected_root_value = end_value_list[0]
        else:
            # Create used function list
            plotter_functions = {"V": self.V.f_n, "e": self.epsilon_n}
            figure = plotter.plot_scalar_field_end_values(
                plotter_functions, parameter_combination, end_value_list, self.settings.scalar_field_domain_plot)
            if not _ipython_info():
                figure.show()
            else:
                plt.show()
            selected_root_value = plotter.ask_scalar_field_end_value(end_value_list, ask_user_defined_point=True)
            if _ipython_info():
                plt.close()

        end_value = numeric_solver.find_root_newton(
            epsilon_minus_one, selected_root_value, fprime, fprime2)

        return end_value

    def _find_N_function(self, parameter_combination, method="numeric", return_value=False, info=True,
                         subprocess_info=False):
        """
        Define process to calculate N(φ) and fint it's inverse function. N is a value which describes inflation scale
        in the end of inflation. It takes an argument of scalar field start value.

        Parameters
        ----------
        parameter_combination : dict
            Combination of parameters
        method : str, optional
            symbolic or numeric, by default "numeric"
        return_value : bool, optional
            Does function returns calculated value?, by default False
        info : bool, optional
            Does function prints information about calculation?, by default True

        Returns
        -------
        bool
            if return_value = True then returns N_function (FunctionType)

        Raises
        ------
        errors.WrongValueError
            In case method != "symbolic" or "numeric".
        """
        # Key for saving result
        key = _create_parameter_key(parameter_combination)

        # Check scalr field domain compatibility
        if self.mode in [1, 3]:
            self.domain_is_continuous(parameter_combination,
                                      self.A.f_s(), self.B.f_s(), self.V.f_s(), self.N_integrand_s())
        else:
            self.domain_is_continuous(parameter_combination, self.IV.f_s(), self.N_integrand_s())

        with np.errstate(divide="ignore", invalid="ignore"):
            if key in self.end_values:
                if info:
                    print("***** Integreerin N-funktsiooni *****")
                    print("***** {} *****".format(key))
                    start_time = time.perf_counter()
                if method == "symbolic" and self._all_functions_symbolic():
                    try:
                        value = self._integrate_N_function_symbolic(
                            parameter_combination, key, info=True, subprocess_info=False)
                        N_symbol = sp.Symbol("N", real=True, positive=True)
                        value = sp.lambdify(N_symbol, value, "numpy")
                    except (TimeoutError, errors.NoSolutionError) as e:
                        print(e)
                        print("Ei saanud sümboolselt lahendada. Lahendan numbriliselt.")
                        value = self._integrate_N_function_numeric(
                            parameter_combination, key, info=True, subprocess_info=False)
                elif method == "numeric":
                    value = self._integrate_N_function_numeric(
                        parameter_combination, key, info=True, subprocess_info=False)
                else:
                    raise errors.WrongValueError(
                        "Method peab olema 'symbolic' või 'numeric'.")

                if info:
                    end_time = time.perf_counter()
                    print("Aeg: {:.2f} s.".format(
                        end_time - start_time))
                    print("***** Leidsin φ(N) funktsiooni. *****")

                self.N_functions[key] = value

                if return_value:
                    return value
            else:
                raise errors.ValueNotDefinedError(
                    "Scalar field end value has not been calculated for this parameter combination.")

    def _integrate_N_function_symbolic(self, parameter_combination, key, info, subprocess_info):
        """
        Calls process to calculate N function inverse symbolically.

        Parameters
        ----------
        parameter_combination : dictionary
            Parameter combination
        key : string
            Key which is used to differentiate solutions.
        info : bool
            Does it print calculation info?
        subprocess_info : bool
            Dose it print φ(N) function

        Returns
        -------
        sp.Expr
            φ(N) function. This function returns scalar field start value for given N.

        Raises
        ------
        errors.NoSolutionError
            Symbolically couldn't find any solution for scalar field end value.
        """
        parameter_combination = _convert_dictionary_strings_to_symbols(
            parameter_combination)
        N_function = self.N_integrand_s().subs(parameter_combination)

        if self.settings.simplify:
            print("Simplifying:")
            N_function = sp.simplify(
                N_function, inverse=self.settings.inverse, rational=self.settings.rational)

        if subprocess_info:
            print("dN(φ) = {}".format(N_function))

        if info:
            print("**** Integreerin funktsiooni ja leian selle pöördfunktsiooni. *****")
            start_time = time.perf_counter()
        N_function_list = symbolic_solver.run_N_fold_integration_symbolic(N_function,
                                                                          self.end_values[key],
                                                                          time=self.settings.timeout)

        if info:
            end_time = time.perf_counter()
            print("Aeg: {:.2f} s.".format(end_time - start_time))
            print("***** Funktsioon integreeritud ja pöördfunktsioon leitud. *****")

        N_function_list = [sp.sympify(
            x, {"N": sp.Symbol("N", real=True, positive=True)}) for x in N_function_list]

        if len(N_function_list) == 0:
            raise errors.NoSolutionError("Ei suutnud leida lahendit {}s. jooksul.".format(self.settings.timeout))
        elif len(N_function_list) == 1:
            N_function = N_function_list[0]
        else:
            figure = plotter.plot_N_function_graphs(N_function_list, self.settings.N_list)
            if _ipython_info():
                figure.show()
            N_function = plotter.ask_right_N_function(N_function_list)
            N_function_numpy = sp.lambdify(sp.Symbol("N", real=True, positive=True), N_function, "numpy")
            if not _ipython_info():
                figure.show()
        if subprocess_info:
            print("φ(N) = {}".format(N_function))

        return N_function_numpy

    def _integrate_N_function_numeric(self, parameter_combination, key, info, subprocess_info=False):
        """
        Calls process to calculate N function inverse numerically.

        Parameters
        ----------
        parameter_combination : dictionary
            Parameter combination
        key : string
            Key which is used to differentiate solutions.
        info : bool
            Does it print calculation info?
        subprocess_info : bool
            Not needed.

        Returns
        -------
        scipy.interp1d
            φ(N) function. This function returns scalar field start value for given N.
        """
        if self._all_functions_symbolic():
            parameter_combination = _convert_dictionary_strings_to_symbols(
                parameter_combination)
            symbolic_function = self.N_integrand_s().subs(parameter_combination)
            if self.settings.simplify:
                print("Simplifying:")
                symbolic_function = sp.simplify(
                    symbolic_function, inverse=self.settings.inverse, rational=self.settings.rational)

            function = sp.lambdify(self._symbol, 1 / symbolic_function, "numpy")
        else:
            def function(x):
                return 1 / self.N_integrand_n(x, **parameter_combination)

        N_function = numeric_solver.integrate_N_fold_numerical(
            function, self.end_values[key], self.settings.N_list)

        return N_function

    def calculate(self, method="numeric", info=True, subprocess_info=False):
        """
        Runs all required functions to calculate parameter values which can be compared with observable values.

        Parameters
        ----------
        method : str, optional
            Is method "numeric" or "symbolic", by default "numeric"
        info : bool, optional
            Does it print calculation info, by default True
        subprocess_info : bool, optional
            Does it print function expressions, by default False
        """
        print("***** Alustan arvutamist *****")
        start_time = time.perf_counter()
        length = 1
        for combination in self.parameter_combinations:
            print("\n***** Kombinatsioon {} *****\n".format(length))
            length += 1
            self.domain_is_valid(combination)
            self._find_field_end_value(
                parameter_combination=combination, method=method, info=info, subprocess_info=subprocess_info)
            self._find_N_function(parameter_combination=combination,
                                  method=method, info=info, subprocess_info=subprocess_info)
        end_time = time.perf_counter()
        print("***** Arvutamine lõpetatud *****")
        print("Programmi töötamise aeg : {:.2f} s.".format(end_time - start_time))

    def plot_graph(self, plot_type=1, plot_id=None, info=True):
        """
        Calls out plotter function for graph plotting.

        Parameters
        ----------
        plot_type : int, optional
            Defines which plot user wants, by default 1
        plot_id : int, optional
            User can define plt.figure number. Used for plotting different models in the same graph, by default None
        info : bool
            Boolean to show all points in ns-r graph (graph_type=1), by default True
        """
        if self.V is None:
            V_func = self.IV.f_n
        else:
            V_func = self.V.f_n

        functions = {"V": V_func, "e": self.epsilon_n, "N": self.N_functions,
                     "r": self.r_n, "ns": self.ns_n
                     }
        plotter.plot(plot_type=plot_type, functions_dict=functions, parameter_combinations=self.parameter_combinations,
                     N_points=self.settings.N_values, N_domain=self.settings.N_list,
                     scalar_field_domain=self.settings.scalar_field_domain_plot,
                     scalar_field_end_values=self.end_values, plot_id=plot_id, model_name=self.name, info=info)

    def plot_show(self):
        plt.show()

    def save_fig(self, name):
        plt.savefig("{}.png".format(name))
