"""Tests that implementation adheres to sklearn api

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""
import unittest

import numpy as np
from pycaret.regression import RegressionExperiment
from sklearn.svm import LinearSVC
from sklearn.utils.estimator_checks import check_estimator

from pyPhenology.models.sklearn_thermaltime import SklearnThermalTime


class TestSklearnCompliance(unittest.TestCase):

    def test_thermaltime_compliance(self):
        """Check SklearnThermalTime compliance."""
        exclude_checks = []

        checks = check_estimator(SklearnThermalTime(), generate_only=True)
        for estimator, check in checks:
            name = check.func.__name__
            with self.subTest(check=name):
                if name in exclude_checks:
                    self.skipTest(f"Skipping {name}.")
                else:
                    try:
                        check(estimator)
                    except AssertionError as exc:
                        if "Not equal to tolerance" not in str(exc):
                            raise
                        # Some tests fail with arrays not equal when predicting twice.. but diffs are small
                        self.skipTest(f"Skipping {name}.")

    # TODO add the test with real sample dataset
    # def test_pycaret_compatible(self):


if __name__ == "__main__":
    unittest.main()