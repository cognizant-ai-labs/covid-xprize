# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

from typing import List

import pandas as pd

from covid_xprize.validation.scenario_generator import ID_COLS, NPI_COLUMNS
from covid_xprize.validation.predictor_validation import _check_columns, _check_geos, _check_days

PRESCRIPTION_INDEX_COL = "PrescriptionIndex"
COLUMNS = ID_COLS + NPI_COLUMNS + ["PrescriptionIndex"]

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}


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
                           dtype={"RegionName": str},
                           error_bad_lines=True)
    ip_df = pd.read_csv(ip_file,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        dtype={"RegionName": str},
                        error_bad_lines=True)

    all_errors = []
    # Check we got the expected columns
    all_errors += _check_columns(set(COLUMNS), presc_df)
    if not all_errors:
        # For each individual prescription in the prescriptions file
        prescription_indexes = presc_df.PrescriptionIndex.unique()
        for i in prescription_indexes:
            i_presc_df = presc_df[presc_df.PrescriptionIndex == i].copy()
            # Columns are good, check we got prescriptions for each requested country / region
            all_errors += _check_geos(ip_df, i_presc_df)
            # Check the IP values
            all_errors += _check_prescription_values(i_presc_df)
            # Check the prediction dates are correct
            all_errors += _check_days(start_date, end_date, i_presc_df)

    return all_errors


def _check_prescription_values(df):
    # For each IP column, check the values are valid
    errors = []
    for ip_name, ip_max_value in IP_MAX_VALUES.items():
        if df[ip_name].isnull().values.any():
            errors.append(f"Column {ip_name} contains NaN values")
        if df[ip_name].min() < 0:
            errors.append(f"Column {ip_name} contains negative values")
        if df[ip_name].max() > ip_max_value:
            errors.append(f"Column {ip_name} contains values higher than max possible value")
    return errors
