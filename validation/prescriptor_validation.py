# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

from typing import List

import pandas as pd

from validation.scenario_generator import ID_COLS, NPI_COLUMNS

PRESCRIPTION_INDEX_COL = "PrescriptionIndex"
COLUMNS = ID_COLS + NPI_COLUMNS + ["PrescriptionIndex"]


def validate_submission(start_date: str,
                        end_date: str,
                        ip_file: str,
                        submission_file: str) -> List[str]:
    """
    Checks a prescription submission file is valid.
    Args:
        start_date: the submission start date as a string, format YYYY-MM-DDD
        end_date: the submission end date as a string, format YYYY-MM-DDD
        ip_file: path to a file-like object
        submission_file: path to a file-like object

    Returns: a list of string messages if errors were detected, an empty list otherwise

    """
    presc_df = pd.read_csv(submission_file,
                           parse_dates=['Date'],
                           encoding="ISO-8859-1",
                           error_bad_lines=True)
    ip_df = pd.read_csv(ip_file,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        error_bad_lines=True)

    all_errors = []
    # Check we got the expected columns
    all_errors += _check_columns(set(COLUMNS), presc_df)
    # if not all_errors:
    #     # Columns are good, check we got prediction for each requested country / region
    #     all_errors += _check_geos(ip_df, pred_df)
    #     # Check the values in PredictedDailyNewCases
    #     all_errors += _check_prediction_values(pred_df)
    #     # Check the prediction dates are correct
    #     all_errors += _check_days(start_date, end_date, pred_df)

    return all_errors


def _check_columns(expected_columns, pred_df):
    errors = []
    missing_columns = expected_columns - set(pred_df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    return errors
