# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

#
# Functions and entrypoint for generating cost csvs to measure prescription cost.
#

import os
import argparse

import numpy as np

from covid_xprize.validation.scenario_generator import get_raw_data
from covid_xprize.validation.scenario_generator import NPI_COLUMNS as IP_COLUMNS

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE = os.path.join(FIXTURES_PATH, "OxCGRT_latest.csv")


def generate_costs(distribution='ones'):
    """
    Returns df of costs for each IP for each geo according to distribution.

    Costs always sum to #IPS (i.e., len(IP_COLUMNS)).

    Available distributions:
        - 'ones': cost is 1 for each IP.
        - 'uniform': costs are sampled uniformly across IPs independently
                     for each geo.
    """
    assert distribution in ['ones', 'uniform'], \
           f'Unsupported distribution {distribution}'


    df = get_raw_data(DATA_FILE, latest=False)

    # Reduce df to one row per geo
    df = df.groupby(['CountryName', 'RegionName']).mean().reset_index()

    # Reduce to geo id info
    df = df[['CountryName', 'RegionName']]

    if distribution == 'ones':
        df[IP_COLUMNS] = 1

    elif distribution == 'uniform':

        # Generate weights uniformly for each geo independently.
        nb_geos = len(df)
        nb_ips = len(IP_COLUMNS)
        samples = np.random.uniform(size=(nb_ips, nb_geos))
        weights = nb_ips * samples / samples.sum(axis=0)
        df[IP_COLUMNS] = weights.T

        # Round weights for better readability with neglible loss of generality.
        df = df.round(2)

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution",
                        type=str,
                        required=True,
                        help="Distribution to generate weights from. Current"
                              "options are 'ones', and 'uniform'.")
    parser.add_argument("-o", "--output_file",
                        type=str,
                        required=True,
                        help="Name of csv file to write generated weights to.")
    args = parser.parse_args()

    print(f"Generating weights with distribution {args.distribution}...")
    weights_df = generate_costs(args.distribution)
    print("Writing weights to file...")
    weights_df.to_csv(args.output_file, index=False)
    print("Done. Thank you.")
