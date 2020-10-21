# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import subprocess
import pandas as pd

CASES_COL = ['NewCases']

PRED_CASES_COL = ['PredictedDailyNewCases']

IP_COLS = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing']

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
    'H3_Contact tracing': 2
}

PREDICT_MODULE = 'examples/lstm/predict.py'
HIST_DATA_FILE_PATH = 'OxCGRT_latest.csv'
TMP_PRED_FILE_NAME = 'tmp_predictions_for_prescriptions/preds.csv'
TMP_PRESCRIPTION_FILE = 'tmp_prescription.csv'


# Function that performs basic preprocessing of historical df
def prepare_df(df):

    # Add GeoID column for easier manipulation
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)

    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing IPs by assuming they are the same as previous day
    for ip_col in IP_MAX_VALUES:
        df.update(df.groupby('GeoID')[ip_col].ffill().fillna(0))

    return df


# Function that wraps predictor in order to query
# predictor when prescribing.
def get_predictions(start_date, end_date, ip_file):

    # Use full path of the local file passed as ip_file
    ip_file_full_path = os.path.abspath(ip_file)

    # Go to covid-xprize root dir to access predict script
    wd = os.getcwd()
    os.chdir("../../..")

    # Run script to generate predictions
    output_str = subprocess.check_output(
        [
            'python', PREDICT_MODULE,
            '--start_date', start_date,
            '--end_date', end_date,
            '--interventions_plan', ip_file_full_path,
            '--output_file', TMP_PRED_FILE_NAME
        ],
        stderr=subprocess.STDOUT
    )

    # Print output from running script
    print(output_str.decode("utf-8"))

    # Load predictions to return
    df = pd.read_csv(TMP_PRED_FILE_NAME)

    # Return to prescriptor dir
    os.chdir(wd)

    return df
