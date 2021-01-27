# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

"""
This is the prescribe.py script for a simple example prescriptor that
generates IP schedules that trade off between IP cost and cases.

The prescriptor is "blind" in that it does not consider any historical
data when making its prescriptions.

The prescriptor is "greedy" in that it starts with all IPs turned off,
and then iteratively turns on the unused IP that has the least cost.

Since each subsequent prescription is stricter, the resulting set
of prescriptions should produce a Pareto front that highlights the
trade-off space between total IP cost and cases.

Note this file has significant overlap with ../random/prescribe.py.
"""

import os
import argparse
import numpy as np
import pandas as pd
from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor
import tempfile
from typing import Union, Callable, Any

NUM_PRESCRIPTIONS = 10

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

IP_COLS = list(IP_MAX_VALUES.keys())


def predict(start_date_str: Union[str, Any], end_date_str: Union[str, Any],
            path_to_hist_file: Union[str, pd.DataFrame],
            prescription: Callable[[str], int] = lambda x: 0) -> pd.DataFrame:

    # Load historical IPs, just to extract the geos
    # we need to prescribe for.
    if isinstance(path_to_hist_file, pd.DataFrame):
        hist_df = path_to_hist_file
    else:
        hist_df = pd.read_csv(path_to_hist_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              keep_default_na=False,
                              error_bad_lines=True)

    # Generate prescriptions
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d') if \
                    isinstance(start_date_str, str) else start_date_str
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d') if \
                    isinstance(end_date_str, str) else end_date_str
    prescription_dict = {
        'CountryName': [],
        'RegionName': [],
        'Date': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []

    #Max prescription
    for country_name in hist_df['CountryName'].unique():
        country_df = hist_df[hist_df['CountryName'] == country_name]
        for region_name in country_df['RegionName'].unique():
                for date in pd.date_range(start_date, end_date):
                    prescription_dict['CountryName'].append(country_name)
                    prescription_dict['RegionName'].append(region_name)
                    prescription_dict['Date'].append(date)
                    for ip in IP_COLS:
                            prescription_dict[ip].append(prescription(ip))

    # Create dataframe from dictionary.
    prescription_df = pd.DataFrame(prescription_dict)

    # Make predictions given all countries

    hist_df = hist_df[hist_df.Date < start_date]
    ips_df = pd.concat([hist_df, prescription_df])
    predictor = XPrizePredictor()
    with tempfile.NamedTemporaryFile() as tmp_ips_file:
        ips_df.to_csv(tmp_ips_file.name)
        return predictor.predict(start_date, end_date, tmp_ips_file.name)