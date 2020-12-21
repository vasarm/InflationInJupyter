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

        self.scalar_field_range = [0, 10]
        self.scalar_field_step = 1e-6
        self.scalar_field_step_plot = 1e-6
        self.scalar_field_domain = np.arange(
            self.scalar_field_range[0], self.scalar_field_range[1]+self.scalar_field_step, self.scalar_field_step)
        self.scalar_field_domain_plot = np.arange(
            self.scalar_field_range[0], self.scalar_field_range[1]+self.scalar_field_step_plot, self.scalar_field_step_plot)


        self.N_values = [50, 60]
        self.N_range = [0, 100]
        self.N_step = 0.1
        self.N_list = np.unique(
            np.concatenate(
                (np.arange(self.N_range[0], self.N_range[1]+self.N_step, self.N_step),
                 self.N_range[1] +
                 np.exp(np.arange(0, 10+self.N_step, self.N_step)),
                 self.N_values)
            )
        )

        # Numerical calculation
        self.derivative_dx = 1e-6

        #
        self.root_precision = 1e-4

        self.simplify = False

    def create_interval_list(self):
        return np.arange(self.scalar_field_range[0], self.scalar_field_range[1]+self.scalar_field_step, self.scalar_field_step)
