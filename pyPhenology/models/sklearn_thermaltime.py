import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd
from .thermaltime import ThermalTime


class SklearnThermalTime(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        t1=None,
        T=None,
        F=None,
        loss_function="rmse",
        method="DE",
        optimizer_params="practical",
        verbose=False,
        debug=False,
    ):
        self.t1 = t1
        self.T = T
        self.F = F
        self.loss_function = loss_function
        self.method = method
        self.optimizer_params = optimizer_params
        self.verbose = verbose
        self.debug = debug
        # TODO check ranges 't1': (-67, 298), 'T': (-25, 25), 'F': (0, 1000)

    def fit(
        self,
        X,
        y,
    ):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        # Define temperature, doy_series
        self.model_ = ThermalTime()

        # Convert incoming data to expected structure as documented here
        # https://pyphenology.readthedocs.io/en/master/data_structures.html

        # year: The year of the temperature timeseries, here we assume that they
        # are in the same year of 2000
        # site_id: A site identifier for each location, here we assume that the
        # number of sites is equal to number of rows in X
        # doy: The julian date of the temperature, here it is a list of numbers
        # each corresponds to a column of X
        # temperature: The temperature i.e. X the predictor

        predictors = pd.DataFrame(
            {
                "year": X.size * [2000],
                "site_id": np.repeat(np.arange(len(X)), X.shape[1]),
                "doy": list(range(X.shape[1])) * X.shape[0],
                "temperature": X.flatten(),
            }
        )
        observations = pd.DataFrame(
            {"year": len(y) * [2000], "site_id": range(len(y)), "doy": y}
        )
        self.model_.fit(
            observations,
            predictors,
            optimizer_params=self.optimizer_params,
            loss_function=self.loss_function,
            method=self.method,
            verbose=self.verbose,
            debug=self.debug,
        )

        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        predictors = {
            "temperature": X.T,
            "doy_series": list(range(X.shape[1]))
            }

        return self.model_.predict(predictors=predictors)
