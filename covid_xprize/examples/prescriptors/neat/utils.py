# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import subprocess
import tempfile
import urllib.request
import pandas as pd
from pathlib import Path

from covid_xprize.validation.scenario_generator import get_raw_data, generate_scenario

# URL for Oxford data
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

# Path to where this script lives
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Data directory (we will download the Oxford data to here)
DATA_PATH = ROOT_DIR / 'data'

# Path to Oxford data file
HIST_DATA_FILE_PATH = DATA_PATH / 'OxCGRT_latest.csv'

# Path to predictor module
PREDICT_MODULE = ROOT_DIR.parent.parent.parent / 'standard_predictor' / 'predict.py'

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
            'H3_Contact tracing',
            'H6_Facial Coverings']

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


def add_geo_id(df):
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
    return df

# Function that performs basic loading and preprocessing of historical df
def prepare_historical_df():

    # Download data if it we haven't done that yet.
    if not os.path.exists(HIST_DATA_FILE_PATH):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        urllib.request.urlretrieve(DATA_URL, HIST_DATA_FILE_PATH)

    # Load raw historical data
    df = pd.read_csv(HIST_DATA_FILE_PATH,
                  parse_dates=['Date'],
                  encoding="ISO-8859-1",
                  error_bad_lines=False)
    df['RegionName'] = df['RegionName'].fillna("")

    # Add GeoID column for easier manipulation
    df = add_geo_id(df)

    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing IPs by assuming they are the same as previous day
    for ip_col in IP_MAX_VALUES:
        df.update(df.groupby('GeoID')[ip_col].ffill().fillna(0))

    return df


# Function to load an IPs file, e.g., passed to prescribe.py
def load_ips_file(path_to_ips_file):
    df = pd.read_csv(path_to_ips_file,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     error_bad_lines=False)
    df['RegionName'] = df['RegionName'].fillna("")
    df = add_geo_id(df)
    return df


# Function that wraps predictor in order to query
# predictor when prescribing.
def get_predictions(start_date_str, end_date_str, pres_df, countries=None):

    # Concatenate prescriptions with historical data
    raw_df = get_raw_data(HIST_DATA_FILE_PATH)
    hist_df = generate_scenario(start_date_str, end_date_str, raw_df,
                                countries=countries, scenario='Historical')
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    hist_df = hist_df[hist_df.Date < start_date]
    ips_df = pd.concat([hist_df, pres_df])

    with tempfile.NamedTemporaryFile() as tmp_ips_file:
        # Write ips_df to file
        ips_df.to_csv(tmp_ips_file.name)

        with tempfile.NamedTemporaryFile() as tmp_pred_file:
            # Run script to generate predictions
            output_str = subprocess.check_output(
                [
                    'python', PREDICT_MODULE,
                    '--start_date', start_date_str,
                    '--end_date', end_date_str,
                    '--interventions_plan', tmp_ips_file.name,
                    '--output_file', tmp_pred_file.name
                ],
                stderr=subprocess.STDOUT
            )

            # Print output from running script
            print(output_str.decode("utf-8"))

            # Load predictions to return
            df = pd.read_csv(tmp_pred_file)

    return df
