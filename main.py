from modules import InflationModel, Settings
import numpy as np
settings = Settings()

model = InflationModel(settings, A="1", B="1", V="x**a")
model.calculate()

