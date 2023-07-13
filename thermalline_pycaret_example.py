#!/usr/bin/env python
# coding: utf-8


from springtime.main import Workflow
import pandas as pd

recipe = "thermaline.yaml"
Workflow.from_recipe(recipe).execute()

df = pd.read_csv("/tmp/output/data.csv")
df = df.set_index(["year", "geometry"])


import pycaret
from pycaret.regression import RegressionExperiment
exp = RegressionExperiment()
exp.setup(df, target = 'breaking leaf buds_doy', session_id = 123)

from pyPhenology.models.sklearn_thermaltime import SklearnThermalTime

result = exp.create_model(SklearnThermalTime(), cross_validation=False)
result
