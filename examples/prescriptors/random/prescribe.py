# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import argparse
import numpy as np
import pandas as pd

NUM_PRESCRIPTIONS = 10

NPI_MAX_VALUES = {
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


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_hist_file: str,
              output_file_path) -> None:

    # Create skeleton df with one row for each prescription
    # for each geo for each day
    hdf = pd.read_csv(path_to_hist_file,
                      parse_dates=['Date'],
                      encoding="ISO-8859-1",
                      error_bad_lines=True)
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    prescription_idxs = []
    country_names = []
    region_names = []
    dates = []

    for prescription_idx in range(NUM_PRESCRIPTIONS):
        for country_name in hdf['CountryName'].unique():
            cdf = hdf[hdf['CountryName'] == country_name]
            for region_name in cdf['RegionName'].unique():
                for date in pd.date_range(start_date, end_date):
                    prescription_idxs.append(prescription_idx)
                    country_names.append(country_name)
                    region_names.append(region_name)
                    dates.append(date.strftime("%Y-%m-%d"))

    prescription_df = pd.DataFrame({
        'PrescriptionIndex': prescription_idxs,
        'CountryName': country_names,
        'RegionName': region_names,
        'Date': dates})

    # Fill df with random values
    num_rows = len(prescription_df)
    for npi_col, max_value in sorted(NPI_MAX_VALUES.items()):
        prescription_df[npi_col] = np.random.randint(0, max_value,
                                                     size=num_rows)

    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save to a csv file
    prescription_df.to_csv(output_file_path, index=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-p", "--prior_ips",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file, args.output_file)
    print("Done!")
