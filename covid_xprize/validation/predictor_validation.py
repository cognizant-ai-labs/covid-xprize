# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.
import argparse
import itertools
from typing import List

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger('predictor_validation')

DATE = "Date"
PREDICTED_DAILY_NEW_CASES = "PredictedDailyNewCases"

COLUMNS = {"CountryName",
           "RegionName",
           "Date",
           PREDICTED_DAILY_NEW_CASES}


def validate_submission(start_date: str,
                        end_date: str,
                        ip_file: str,
                        submission_file: str) -> List[str]:
    """
    Checks a prediction submission file is valid.
    Args:
        start_date: the submission start date as a string, format YYYY-MM-DDD
        end_date: the submission end date as a string, format YYYY-MM-DDD
        ip_file: path to a file-like object
        submission_file: path to a file-like object

    Returns: a list of string messages if errors were detected, an empty list otherwise

    """
    pred_df = pd.read_csv(submission_file,
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          dtype={"RegionName": str},
                          error_bad_lines=True)
    ip_df = pd.read_csv(ip_file,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        dtype={"RegionName": str},
                        error_bad_lines=True)

    all_errors = []
    # Check we got the expected columns
    all_errors += _check_columns(COLUMNS, pred_df)
    if not all_errors:
        # Columns are good, check we got prediction for each requested country / region
        all_errors += _check_geos(ip_df, pred_df)
        # Check the values in PredictedDailyNewCases
        all_errors += _check_prediction_values(pred_df)
        # Check the prediction dates are correct
        all_errors += _check_days(start_date, end_date, pred_df)

    return all_errors


def _check_columns(expected_columns, pred_df):
    errors = []
    # Make sure each column is present
    missing_columns = expected_columns - set(pred_df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
        # Columns are not there, can't check anything more
        return errors

    # Make sure column Date contains dates
    date_column_type = pred_df[DATE].dtype
    if not np.issubdtype(date_column_type, np.datetime64):
        errors.append(f"Column {DATE} contains non date values: {date_column_type}")

    # Make sure column PredictedDailyNewCases contains numbers
    cases_column_type = pred_df[PREDICTED_DAILY_NEW_CASES].dtype
    if not np.issubdtype(cases_column_type, np.number):
        errors.append(f"Column {PREDICTED_DAILY_NEW_CASES} contains non numerical values: {cases_column_type}")

    return errors


def _check_prediction_values(df):
    # Make sure the column containing the predictions is there
    errors = []
    if PREDICTED_DAILY_NEW_CASES in df.columns:
        if df[PREDICTED_DAILY_NEW_CASES].isnull().values.any():
            errors.append(f"Column {PREDICTED_DAILY_NEW_CASES} contains NaN values")
        if any(df[PREDICTED_DAILY_NEW_CASES] < 0):
            errors.append(f"Column {PREDICTED_DAILY_NEW_CASES} contains negative values")
    return errors


def _check_geos(ip_df, pred_df):
    errors = []
    _add_geoid_column(ip_df)
    _add_geoid_column(pred_df)
    requested_geo_ids = set(ip_df.GeoID.unique())
    actual_geo_ids = set(pred_df.GeoID.unique())
    # Check if any missing
    # Additional geos are OK, but predictions should at least include requested ones
    missing_geos = requested_geo_ids - actual_geo_ids
    if missing_geos:
        errors.append(f"Missing countries / regions: {missing_geos}")
    return errors


def _add_geoid_column(df):
    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    df["GeoID"] = np.where(df["RegionName"].isnull(),
                           df["CountryName"],
                           df["CountryName"] + ' / ' + df["RegionName"])


def _check_days(start_date, end_date, df):
    errors = []
    _add_geoid_column(df)
    # Sort by geo and date
    df.sort_values(by=["GeoID", "Date"], inplace=True)
    # Convert the dates
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    num_days = (end_date - start_date).days + 1
    expected_dates = [start_date + pd.offsets.Day(i) for i in range(num_days)]
    # Get the geo names
    geo_ids = list(df.GeoID.unique())
    for geo_id in geo_ids:
        pred_dates = df[df.GeoID == geo_id].Date
        for expected_date, pred_date in itertools.zip_longest(expected_dates, pred_dates, fillvalue=None):
            if not expected_date == pred_date:
                errors.append(f"{geo_id}: Expected prediction for date "
                              f"{expected_date.strftime('%Y-%m-%d') if expected_date is not None else None}"
                              f" but got {pred_date.strftime('%Y-%m-%d') if pred_date is not None else None}")
    return errors


def do_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=False,
                        default="2020-12-22",
                        help="Start date from which to apply the scenario"
                             "Format YYYY-MM-DD. For example 2020-12-22")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=False,
                        default="2021-06-19",
                        help="Last date of the scenario"
                             "Format YYYY-MM-DD. For example 2021-06-19")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-f", "--submission_file",
                        dest="submission_file",
                        type=str,
                        required=True,
                        help="Path to the filename containing the submission (predictions) to be validated.")
    args = parser.parse_args()

    submission_file = args.submission_file
    start_date = args.start_date
    end_date = args.end_date
    ip_file = args.ip_file
    LOGGER.info(f"Validating submissions file {submission_file} "
                f"start date {start_date} end date {end_date} intervention plan {ip_file}")

    errors = validate_submission(start_date, end_date, ip_file, submission_file)
    if not errors:
        LOGGER.info(f'{submission_file} submission passes validation')
    else:
        LOGGER.warning(f'Submission {submission_file} has errors: ')
        LOGGER.warning('\n'.join(errors))

    LOGGER.info(f"Done!")


if __name__ == '__main__':
    do_main()
