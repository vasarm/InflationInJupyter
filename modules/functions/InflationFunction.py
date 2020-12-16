import InflationInJupyter.modules.functions.InflationFunctionBase
from .. import errors as errors
from .. import config as config

class InflationFunction(InflationFunctionBase):

    def __init__(self, function, name, settings, **kwargs):
        super().__init__(function, **kwargs)
        self.name = name
        self.settings = settings

    @property
    def name(self):
        return __name

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
                raise errors.WrongValueError("Name can't be empty (name != '').")
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
        return self.symbolic_function()

    def n_s(self, x):
        """
        Returns numeric function values
        """
        return self.numeric_function(x)

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
        return self.numeric_derivative(x, dx=self.settings.dx, **kwargs)

    def dd_n(self, x, **kwargs):
        """
        Return function second derivative values at point x.
        """
        return self.numeric_second_derivative(x, dx=self.settings.dx, **kwargs)