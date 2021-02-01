import numpy as np


class Settings:
    """
    Class for different program settings. At the start these settings are loaded from predefined file.
    """

    def __init__(self):
        """
        This function has different properties:

        Attributes
        ----------
        self.derivative_dx : float
            Increment size for derivative calculation.

        self.formalism : str
            Possible two values. Palatini or Metric, by default None.
        """
        self.formalism = "Metric"

        self.scalar_field_step = 1e-6
        self.scalar_field_range = [0, 10]

        self.scalar_field_step_plot = 1e-5
        self.scalar_field_domain = np.arange(
            self.scalar_field_range[0], self.scalar_field_range[1] + self.scalar_field_step, self.scalar_field_step)
        self.scalar_field_domain_plot = np.arange(
            self.scalar_field_range[0], self.scalar_field_range[1] + self.scalar_field_step_plot,
            self.scalar_field_step_plot)

        self.N_values = [50, 60]
        self.N_range = [0, 100]
        self.N_step = 0.1
        self.N_list = np.unique(
            np.concatenate(
                (np.arange(self.N_range[0], self.N_range[1] + self.N_step, self.N_step),
                 self.N_range[1] +
                 np.exp(np.arange(0, 10 + self.N_step, self.N_step)),
                 self.N_values)
            )
        )

        # key - scalar field values.
        self.find_V_intervals = {}
        self.find_V_increment = 1e-6
        # Numerical calculation
        self.derivative_dx = 1e-6

        #
        self.root_precision = 1e-4

        # simplify
        self.simplify = False
        self.simplify_inverse = True
        self.simplify_rational = True

        # Function display
        self.display = "latex"

        # Set time limet (in seconds) for symbolic calculations
        self.timeout = 10

        # Add small number to divisor so it will avoid division by zero
        self.divisor_inc = 1e-100
        # During calculation does it use symbolic function for numeri calculation
        self.symbolic_to_numeric = False

        # Simplify settings
        self.inverse = True
        self.rational = True

    @property
    def scalar_field_range(self):
        return self.__scalar_field_range

    @scalar_field_range.setter
    def scalar_field_range(self, new_value):
        if isinstance(new_value, list) and len(new_value) == 2 and new_value[0] < new_value[1]:
            self.__scalar_field_range = new_value
        else:
            raise TypeError("Wrong type value as scalar field range.")
        self.scalar_field_domain = np.arange(
            self.scalar_field_range[0], self.scalar_field_range[1] + self.scalar_field_step, self.scalar_field_step)

    def create_interval_list(self):
        return np.arange(self.scalar_field_range[0], self.scalar_field_range[1] + self.scalar_field_step,
                         self.scalar_field_step)

    def find_V_domain(self, key):
        return np.arange(self.find_V_intervals[key][0], self.find_V_intervals[key][1] + self.find_V_increment,
                         self.find_V_increment)
