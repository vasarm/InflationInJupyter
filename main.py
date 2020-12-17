if __name__ == "__main__":
    from modules.model import InflationModel
    from modules.config import Settings

    import numpy as np
    settings = Settings()

    model = InflationModel(settings, A="1", B="1", V="x**a")
    model.calculate()

    def test(x):
        """
        Testing

        Parameters
        ----------
        x : array_like
            Test

        Returns
        -------
        [type]
            [description]
        """
        return x