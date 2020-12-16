import numpy as np


class Settings:
    """
    Class for different program settings. At the start these settings are loaded from according file.
    """

    def __init__(self):

        self.formalism = "Metric"

        self.scalar_field_range = [0, 10]
        self.scalar_field_step = 1e-7

        self.N_range = [0, 100]
        self.N_step = 0.1
        self.N_list = np.arange(self.N_range[0], self.N_range[1]+self.N_step, self.N_step)

        # Numerical calculation
        self.derivative_dx = 1e-6

        #
        self.root_precision = 1e-4
        
        self.simplify = False

        """
        This function has different properties:
        derivative_dx : float
            Increment size for derivative calculation.

        formalism : str
            Possible two values. Palatini or Metric, by default None.
        """

    def create_interval_list(self):
        return np.arange(self.scalar_field_range[0], self.scalar_field_range[1]+self.scalar_field_step, self.scalar_field_step)
