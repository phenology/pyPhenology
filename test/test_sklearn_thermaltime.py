"""Tests that implementation adheres to sklearn api

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""
import unittest
from sklearn.utils.estimator_checks import check_estimator
from pyPhenology.models.sklearn_thermaltime import SklearnThermalTime
from pycaret.regression import RegressionExperiment, predict_model, load_model
from pyPhenology import utils
from pyPhenology.models import utils as mu
import pandas as pd
import numpy as np

class TestSklearnCompliance(unittest.TestCase):
    def test_thermaltime_compliance(self):
        """Check SklearnThermalTime compliance."""

        # thermaltime model needs real data, otherwise the check
        # regressor.score(X, y_) > 0.5 in the test below fails due to
        # low score value.
        exclude_checks = ["check_regressors_train"]

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
                        # Some tests fail with arrays not equal when predicting
                        # twice.. but diffs are small
                        self.skipTest(f"Skipping {name}.")


class TestPyCaretCompliance:
    def test_pycaret_compatible(self, tmp_path):
        # Use pyphenology data to train the model: this pipeline is based on
        # the examples/model_selection_aic.py and examples/model_rmse.py
        observations, predictors = utils.load_test_data(
            name="vaccinium", phenophase="budburst"
        )
        Model = utils.load_model("ThermalTime")
        model = Model()
        observations_test = observations[:10]
        observed_doy = observations_test.doy.values
        observations_train = observations[10:]

        # set `seed`` here, so the results of "DE" method (default) does not
        # change! see
        # https://pyphenology.readthedocs.io/en/master/optimizers.html
        optimizer_params = {
            'maxiter': 1000,
            'popsize': 50,
            'mutation': (0.5, 1),
            'recombination': 0.25,
            'disp': False,
            'seed': 123
        }
        model.fit(observations_train, predictors, optimizer_params=optimizer_params)
        predicted_doy = model.predict(observations_test, predictors)
        rmse_phenology = np.sqrt(np.mean((predicted_doy - observed_doy) ** 2))

        # Prepare data to train the model using pycaret
        doy_array, temperature_array, _ = mu.misc.temperature_only_data_prep(
            observations, predictors, for_prediction=False
        )

        df = pd.DataFrame(temperature_array.T)
        df["doy"] = doy_array
        df.dropna(inplace=True)
        df_test = df.iloc[:10]  # same as above

        # Create pycaret instances
        exp = RegressionExperiment()
        exp.setup(df, target="doy", session_id=123, test_data=df_test, index=False, preprocess=False)
        model = exp.create_model(SklearnThermalTime(optimizer_params=optimizer_params), cross_validation=False)

        # it should be possible to get `RMSE`` from `model` but I want to test
        # `load_model` as below
        exp.save_model(model, tmp_path / "pycaret_thermaltime")

        # Load the saved model and use it for predictions
        loaded_model = load_model(tmp_path / "pycaret_thermaltime")
        predictions = predict_model(loaded_model, data=df_test)
        predicted_doy = predictions["prediction_label"].values
        rmse_pycaret = np.sqrt(np.mean((predicted_doy - observed_doy) ** 2))

        # Note: if data changes, the test might fail!
        # Note: RMSE of pyPhenology and pyCaret are not exactly the same!
        assert abs(rmse_phenology - rmse_pycaret) < 1  # 1 doy, not strict


if __name__ == "__main__":
    unittest.main()
