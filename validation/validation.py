import itertools
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

PREDICTED_DAILY_NEW_CASES = "PredictedDailyNewCases"

COLUMNS = ["CountryName",
           "RegionName",
           "Date",
           PREDICTED_DAILY_NEW_CASES]


def validate_submission(start_date: str, end_date: str, submission_url: str) -> Optional[List[str]]:
    """
    Checks a submission file is valid.
    Args:
        start_date: the submission start date as a string, format YYYY-MM-DDD
        end_date: the submission end date as a string, format YYYY-MM-DDD
        submission_url: path to a file-like object

    Returns: a list of string messages if errors were detected, None otherwise

    """
    df = pd.read_csv(submission_url,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     error_bad_lines=True)

    all_errors = []
    # Check we go the expected columns
    errors = _check_columns(COLUMNS, df)
    if errors:
        all_errors.extend(errors)
    else:
        # Columns are good, check the values in PredictedDailyNewCases
        errors = _check_prediction_values(df)
        if errors:
            all_errors.extend(errors)
        # Now check the prediction dates are correct
        errors = _check_days(start_date, end_date, df)
        if errors:
            all_errors.extend(errors)

    if all_errors:
        return all_errors
    else:
        return None


def _check_columns(expected_columns, df):
    if not expected_columns == list(df.columns):
        return [f"Not the expected list of columns. Expected columns are: {expected_columns}"]
    return None


def _check_prediction_values(df):
    # Make sure the column containing the predictions is there
    errors = []
    if PREDICTED_DAILY_NEW_CASES in df.columns:
        if df[PREDICTED_DAILY_NEW_CASES].isnull().values.any():
            errors.append(f"Column {PREDICTED_DAILY_NEW_CASES} contains NaN values")
        if any(df[PREDICTED_DAILY_NEW_CASES] < 0):
            errors.append(f"Column {PREDICTED_DAILY_NEW_CASES} contains negative values")
        return errors
    return None


def _check_days(start_date, end_date, df):
    errors = []
    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    df["GeoID"] = np.where(df["RegionName"].isnull(),
                           df["CountryName"],
                           df["CountryName"] + ' / ' + df["RegionName"])
    # Sort by geo and date
    df.sort_values(by=["GeoID", "Date"], inplace=True)
    # Convert the dates
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    num_days = (end_date - start_date).days + 1
    expected_dates = [start_date + pd.offsets.Day(i) for i in range(num_days)]
    # Get the geo names
    geoids = list(df.GeoID.unique())
    for geoid in geoids:
        pred_dates = df[df.GeoID == geoid].Date
        for expected_date, pred_date in itertools.zip_longest(expected_dates, pred_dates, fillvalue=None):
            if not expected_date == pred_date:
                if expected_date is None:
                    errors.append(f"{geoid}: Was not expecting prediction for {pred_date}")
                else:
                 errors.append(f"{geoid}: Expected prediction for date {expected_date} but got {pred_date}")
    if errors:
        return errors
    else:
        return None
