class Settings:
    """
    Class for different program settings. At the start these settings are loaded from according file.
    """
    def __init__(self):

        self.formalism = "Metric"


        # Numerical calculation
        self.dx = 1e-6

        """
        This function has different properties:
        dx : float
            Increment size for derivative calculation.

        formalism : str
            Possible two values. Palatini or Metric, by default None.
        
        """