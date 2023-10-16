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

        # Fit the model
        self.model_ = ThermalTime()
        self.model_.fit(
            y,
            X,
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

        return self.model_.predict(predictors=X)
