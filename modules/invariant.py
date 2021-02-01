import numpy as np
import sympy as sp

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

import time

import modules.errors as errors
from modules.config import Settings
from modules.function import FunctionA, FunctionB, FunctionV, FunctionIV, FunctionIVf, FunctionIF, FunctionIFf
import modules.symbolicfunctions as symbolic_solver
import modules.numericfunctions as numeric_solver
import modules.plotter as plotter


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


class InvariantSystem:
    """
    Class where combination of functions are defined and unknown function is calculated.
    For more look https://arxiv.org/abs/2005.14571
    """

    def __init__(self, model):
        """
        Initializes InvariantSystem. Create three new functions which are needed in certain cases.
        """
        if model.mode == 0:
            raise errors.ModeError("Mode must be other than 0. (1, 2 or 3)")
        self.model = model
        self.settings = model.settings
        self.A = model.A
        self.B = model.B
        self.V = model.V
        self.IV = model.IV

        self._symbol = sp.Symbol("x", real=True)

        # Function IV(φ) in other words, invariant potential which variable is scalar field
        self.IVf = None
        # Function Iφ(IV) in other words, invariant scalr field which variable is invariant potential
        self.IF = None
        # Function Iφ(φ) in other words, invariant scalar field which variable is scalar field
        self.IFf = None

    def _display_function(self, function):
        if self.settings.display == "latex":
            print(str(sp.latex(function)))
        else:
            print(function)

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

        # create V function calculation fully (numerical and symbolical.)

        # implement B function calculation

    @property
    def IVf(self):
        return self.__IVf

    @IVf.setter
    def IVf(self, IVf):
        if isinstance(IVf, FunctionIVf):
            self.__IVf = IVf
        elif isinstance(IVf, str):
            self.__IVf = FunctionIVf(IVf, self.settings)
        elif IVf is None:
            self.__IVf = None
        else:
            raise errors.WrongTypeError(
                "Function IV(φ) must be FunctionIVf type.")

    @property
    def IF(self):
        return self.__IF

    @IF.setter
    def IF(self, IF):
        if isinstance(IF, FunctionIF):
            self.__IF = IF
        elif isinstance(IF, str):
            self.__IF = FunctionIF(IF, self.settings)
        elif IF is None:
            self.__IF = None
        else:
            raise errors.WrongTypeError("Function IF must be FunctionIF type.")

    @property
    def IFf(self):
        return self.__IFf

    @IFf.setter
    def IFf(self, IFf):
        if isinstance(IFf, FunctionIFf):
            self.__IFf = IFf
        elif isinstance(IFf, str):
            self.__IFf = FunctionIFf(IFf, self.settings)
        elif IFf is None:
            self.__IFf = None
        else:
            raise errors.WrongTypeError(
                "Function IFf must be FunctionIFf type.")

    @property
    def settings(self):
        return self.__settings

    @settings.setter
    def settings(self, new_settings):
        if isinstance(new_settings, Settings):
            self.__settings = new_settings
        else:
            raise errors.WrongTypeError(
                "Settings must be defined and Settings type.")

    def find_IF(self, method="symbolic", info=True):
        """
        General function to start calculating IV inverse.

        Parameters
        ----------
        method : str, optional
            "symbolic" or "numeric", by default "symbolic"
        info : bool, optional
            Does program display info for user., by default True

        Raises
        ------
        errors.MethodError
            This function currently works only symbolically. Otherwise raises error.
        errors.FunctionNotDefinedError
            To find inverse IV function must be defined.
        """
        if method != "symbolic":
            raise errors.MethodError(
                "Numeric method is not defined yey due to it's complexity.")

        if self.IV is None:
            raise errors.FunctionNotDefinedError(
                "IV function must be defined to calculate Iφ(IV)")

        if info:
            print("***** Looking for Iφ(IV) function *****")
            start_time = time.perf_counter()

        IF_sol = self._find_IF_symbolic()
        if info:
            end_time = time.perf_counter()
            print("***** Iφ(IV) function found *****")
            print("Time taken: {:.2f} s.".format(
                end_time - start_time))
        self.IF = FunctionIF(IF_sol, self.settings)

    def _find_IF_symbolic(self):
        """
        Find function IF (Iφ(IV)) symbolically.

        Returns
        -------
        FucntionIF
            Returns function IF (Iφ(IV)).

        Raises
        ------
        errors.NoSolutionError
            When sympy can't find IV function inverse.
        """
        IF_sym_solutions = symbolic_solver.run_IF_calculation_symbolical(
            self.IV.f_s(), time=10)

        if len(IF_sym_solutions) == 0:
            raise errors.NoSolutionError(
                "Program could not find inverse function.")
        elif len(IF_sym_solutions) == 1:
            return IF_sym_solutions[0]
        else:
            print("***** More than one solution *****")
            for order, function in enumerate(IF_sym_solutions, 1):
                print("{}. Function : ".format(order))
                self._display_function(function)

            def select_solution():
                user_input = int(input("Choose the solution:"))
                try:
                    return IF_sym_solutions[user_input - 1]
                except:
                    print(
                        "Error occurred. Enter correct value. ({} - {})?".format("1", len(IF_sym_solutions)))
                    return select_solution()

            return select_solution()

    def find_A(self, method="numeric", info=True):
        """
        Executes progress, to calculate V. Only method="numeric" possible.

        Parameters
        ----------
        method : str, optional
            Only "numeric" possible, by default "numeric"
        info : bool, optional
            Bool for printing progress, by default True

        Raises
        ------
        errors.FunctionNotDefinedError
            [description]
        """
        if any([x is None for x in [self.A, self.B, self.IV]]):
            raise errors.FunctionNotDefinedError("Functions A, B and IV must be defined to find V.")
        if info:
            print("***** Looking for function A(φ) *****")
            start_time = time.perf_counter()


        # symbolic solution not available, as it is a bit complex
        solution = self.find_A_numerical()

        self.A = FunctionA(solution, self.settings)
        self.model.a = self.a

    def find_A_numerical():
        pass



    def find_B(self, method="symbolic", info=True):
        """
        Executes progress, to calculate V. Only method="symbolic" possible as it ise 100% solvable analytically.
        And numerical function can be defined by symbolic function.

        Parameters
        ----------
        method : str, optional
            Only "symbolic" possible, by default "symbolic"
        info : bool, optional
            Bool for printing progress, by default True

        Raises
        ------
        errors.FormalismError
            Formalism must be "Metric" or "Palatini"
        """
        if any([x is None for x in [self.A, self.V, self.IV]]):
            raise errors.FunctionNotDefinedError("Functions A, V and IV must be defined to find V.")

        if self.IF is None:
            self.find_IF(info=info)

        if info:
            print("***** Looking for B(φ) function *****")
            start_time = time.perf_counter()
        if method == "symbolic":
            B_solution = self.find_B_symbolical(info)

        else:
            raise errors.MethodError("Method must be 'symbolic'.")

        if info:
            end_time = time.perf_counter()
            print("***** Iφ(IV) function found *****")
            print("Time taken: {:.2f} s.".format(
                end_time - start_time))

        self.B = FunctionB(B_solution, self.settings)
        self.model.B = FunctionB(B_solution, self.settings)

    def find_B_symbolical(self, info):
        """
        Symbolical way for calculating B.

        Parameters
        ----------
        info : bool
            Does it print progress information.

        Returns
        -------
        sp.Expr
            Found function B.

        Raises
        ------
        errors.FormalismError
            [description]
        """

        # Calculate dIφ/dφ = dIφ/dIv * dIv/dφ
        IFderivative = self.IF.d_s().subs(self._symbol, self.V.f_s() / self.A.f_s() ** 2)
        IVfderivative = sp.diff(self.V.f_s() / self.A.f_s() ** 2, self._symbol)
        IFfderivative = IFderivative * IVfderivative

        if self.settings.formalism == "Metric":
            B = self.A.f_s() * (IFfderivative ** 2 - 3 / 2 * (self.A.d_s() / self.A.f_s()) ** 2)
        elif self.settings.formalism == "Palatini":
            B = self.A.f_s() * IFfderivative ** 2
        else:
            raise errors.FormalismError("Formalism must be 'Palatini' or 'Metric'")

        return B

    def find_V(self, method="numeric", info=True):
        """
        Executes progress, to calculate V. Only method="numeric" possible.

        Parameters
        ----------
        method : str, optional
            Only "numeric" possible, by default "numeric"
        info : bool, optional
            Bool for printing progress, by default True

        Raises
        ------
        errors.FunctionNotDefinedError
            [description]
        """
        if any([x is None for x in [self.A, self.B, self.IV]]):
            raise errors.FunctionNotDefinedError("Functions A, B and IV must be defined to find V.")
        if info:
            print("***** Looking for function V(φ) *****")
            start_time = time.perf_counter()


        # symbolic solution not available, as it is a bit complex
        solution = self.find_V_numerical()

        self.V = FunctionV(solution, self.settings)
        self.model.V = self.V



    def find_V_numerical(self):
        """
        Numerical calculation for V function. More precisely this function defines function, which defines V function
        as it depends on parameter values and can't analytically derivate it.
        """

        def V_function(x, **kwargs):

            key = _create_parameter_key(kwargs)
            print("***** Määra skalaarvälja alguspunkt *****")

            def ask_start():
                try:
                    start = sp.sympify(input("Enter start value: "))
                except:
                    return ask_start()
                if start.is_number and start < self.settings.scalar_field_range[1]:
                    return float(start)
                else:
                    return ask_start()

            start_value = ask_start()
            self.settings.scalar_field_range = [start_value, self.settings.scalar_field_range[1]]
            x = np.array(self.settings.scalar_field_domain)
            print("***** Määra potentsiaali väärtus kohal φ={:.6f} *****".format(start_value))

            def ask_V0_value():
                try:
                    value = np.sympify(input("Enter V_0 value: "))
                except:
                    return ask_start()
                if value.is_number:
                    return float(value)
                else:
                    return ask_start()

            V0 = ask_V0_value()

            function = lambda x: self.IV.f_n(x, **kwargs) - V0/self.A.f_n(start_value, **kwargs)**2

            def find_roots(function, x_interval):
                # Find places where it changes from positive to negative or other way
                y = function(x_interval)
                sign = np.sign(y)
                zeros = (np.roll(sign, 1) - sign) != 0
                zeros = x_interval[zeros]
                zeros = np.array([newton(func=function, x0=zero) for zero in zeros])
                return zeros

            probable_values = find_roots(function, np.linspace(-100, 100, 10000))
            if len(probable_values) == 0:
                raise errors.NoSolutionError(
                    "No end value found. Changing interval might help.")
            elif len(probable_values) == 1:
                selected_root_value = probable_values[0]
            else:
                print("Vali sobiv väärtus väärtus.")
                for num, elem in enumerate(probable_values, 1):
                    print("{}. φ={:.6f}".format(num, elem))

                def ask_right_value():
                    select_value = int(input("Sisesta valitud väärtus: "))
                    try:
                        return probable_values[select_value-1]
                    except:
                        return ask_right_value()

                selected_root_value = ask_right_value()
            if self.settings.formalism == "Metric":
                integrand = lambda x: np.sqrt(self.B.f_n(x, **kwargs)/self.A.f_n(x,**kwargs) - 3/2 * (self.A.d_n(x, **kwargs)/self.A.f_n(x, **kwargs))**2)
            elif self.settings.formalism == "Palatini":
                integrand = lambda x: np.sqrt(self.B.f_n(x, **kwargs) / self.A.f_n(x, **kwargs))

            def ode_function(y, t):
                return integrand(t)
            I_F_fii = np.reshape(odeint(func=ode_function, y0=selected_root_value, t=self.settings.scalar_field_domain), -1)
            # Calculate integral I_φ(φ)
            IVfii = self.IV.f_n(x=I_F_fii, **kwargs)
            # AS looking for IFf might change the domain it changes it automatically in settings
            # NB! Problem may occur, if user later calls again this function but wrong scalar field values are used.
            # x = self.settings.find_V_domain(_create_parameter_key(kwargs))
            V = IVfii * self.A.f_n(x=x, **kwargs) ** 2
            V_func = interp1d(x, V)
            return V_func

        return V_function
