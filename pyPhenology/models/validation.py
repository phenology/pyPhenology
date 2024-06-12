import pandas as pd
from sklearn.utils import check_X_y, check_array


def validate_sklearn_Xy(predictors, observations):
    """Input validation for standard estimators.

    check_X_y will raise a ValueError if the inputs are not compatible."""
    valid_sklearn_Xy = True
    try:
        predictors, observations = check_X_y(predictors, observations)
    except ValueError:
        valid_sklearn_Xy = False

    return valid_sklearn_Xy, predictors, observations


def validate_sklearn_X(predictors):
    """Input validation for standard estimators.

    check_array will raise a ValueError if the inputs are not compatible."""
    valid_sklearn_X = True
    try:
        predictors = check_array(predictors)
    except ValueError:
        valid_sklearn_X = False

    return valid_sklearn_X, predictors


def validate_predictors(predictor_df, valid_columns):
    """ Validate the required columns in a predictor dataframe

    Parameters
    ----------
    predictor_df : Pandas Dataframe

    Returns
    -------
    predictor_df : The same dataframe but with only the valid columns
    """
    if not isinstance(predictor_df, pd.DataFrame):
        raise TypeError('predictors should be a pandas dataframe')
    for column in valid_columns:
        if column not in predictor_df.columns:
            raise ValueError('missing required predictor columns: ' + column)

    return predictor_df[valid_columns]


def validate_observations(observations, for_prediction=False):
    """ Validate an observations dataframe to the format used in this package.

    Parameters
    ----------
    observations : Pandas Dataframe

    for_prediction : bool
        If being used to in model.predict(), then one less column is required

    Returns
    -------
    observations : The same dataframe but with only the valid columns
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError('observations should be a pandas dataframe')
    valid_columns = ['year', 'site_id']
    if not for_prediction:
        valid_columns.append('doy')

    for column in valid_columns:
        if column not in observations.columns:
            raise ValueError('missing required observations column: ' + column)
        if observations[column].isna().any():
            raise ValueError('Nan values in observation column: ' + column)

    return observations[valid_columns]


def validate_model(model_class):
    required_attributes = ['_apply_model', 'all_required_parameters', '_required_data',
                           '_organize_predictors', '_validate_formatted_predictors']
    for attribute in required_attributes:
        if not hasattr(model_class, attribute):
            raise RuntimeError('Missing model attribute: ' + str(attribute))
