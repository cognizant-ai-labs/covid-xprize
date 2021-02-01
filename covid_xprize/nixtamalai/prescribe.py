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
# from covid_xprize.examples.prescriptors.neat.utils import PRED_CASES_COL, prepare_historical_df, CASES_COL, IP_COLS, \
#    IP_MAX_VALUES, add_geo_id, get_predictions
from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor
from covid_xprize.nixtamalai import surrogate_model
from covid_xprize.nixtamalai.helpers import add_geo_id
from microtc.utils import load_model
import tempfile
from os import path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


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
IP_COLS.sort()


def prescriptions(dd):
    prescriptions_path = path.join(ROOT_DIR,
                                   "2021-01-28-prescriptions", 
                                   "%s.pickle.gz")
    add_geo_id(dd)
    dd.set_index("GeoID", inplace=True)
    dd.sort_index(inplace=True)
    regions_id = {v: k for k, v in enumerate(dd.index)}
    output = dict()
    for geoid in dd.index:
        w = dd.loc[geoid, IP_COLS].values
        prescriptions = load_model(prescriptions_path % regions_id[geoid])
        cost = {k: [v, (np.array([int(i) for i in k]) * w).sum()] for k, v in prescriptions.items()}
        npis = list(cost.keys())
        npis.sort(key=lambda x: cost[x][0])
        _ = np.array([cost[k] for k in npis])
        index = surrogate_model.is_pareto_efficient(_, return_mask=False)
        if index.shape[0] > 10:
            ind2 = np.linspace(1, index.shape[0] - 2, 10).round().astype(np.int)
            ind2 = index[ind2]
        else:
            ind2 = index
        _ = [npis[x] for x in ind2]
        output.update({geoid: _})
    return output


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_hist_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    # Load historical IPs, just to extract the geos
    # we need to prescribe for.
    hist_df = pd.read_csv(path_to_hist_file,
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          keep_default_na=False,
                          error_bad_lines=True)
    add_geo_id(hist_df)

    # Load the IP weights, so that we can use them
    # greedily for each geo.
    weights_df = pd.read_csv(path_to_cost_file, keep_default_na=False)
    presc = prescriptions(weights_df)

    # Generate prescriptions
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    prescription_dict = {
        'CountryName': [],
        'RegionName': [],
        'Date': [],
        'PrescriptionIndex': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []

    for geoid, df in hist_df.groupby("GeoID"):
        country_name = df.iloc[0].CountryName
        region_name = df.iloc[0].RegionName
        data = presc[geoid]
        if len(data) < NUM_PRESCRIPTIONS:
            data += [data[0] for _ in range(len(data), NUM_PRESCRIPTIONS)]
        for prescription_idx, prescriptor in enumerate(data):
            for date in pd.date_range(start_date, end_date):
                date_str = date.strftime("%Y-%m-%d")
                prescription_dict['CountryName'].append(country_name)
                prescription_dict['RegionName'].append(region_name)
                prescription_dict['Date'].append(date_str)
                prescription_dict['PrescriptionIndex'].append(prescription_idx)
                for npi, value in zip(IP_COLS, prescriptor):
                    prescription_dict[npi].append(int(value))

    prescription_df = pd.DataFrame(prescription_dict)

    # Create the directory for writing the output file, if necessary.
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save output csv file.
    prescription_df.to_csv(output_file_path, index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file)
    print("Done!")
