# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

#
# Functions and entrypoint for generating cost csvs to measure prescription cost.
#

import os
import argparse

import numpy as np
import pandas as pd

from covid_xprize.validation.scenario_generator import NPI_COLUMNS as IP_COLUMNS

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GEOS = os.path.join(ROOT_DIR, '..', '..', "countries_regions.csv")


def generate_costs(distribution='ones'):
    """
    Returns a df of costs for each IP for default list of geos according to distribution.
    """
    return generate_costs_for_geos_file(DEFAULT_GEOS, distribution)


def generate_costs_for_geos_file(geos_file, distribution='ones'):
    """
    Returns a df of costs for each IP for geos in geos_file according to distribution.
    """
    geos_df = load_geos(geos_file)
    return generate_costs_for_geos_df(geos_df, distribution)


def generate_costs_for_geos_df(geos_df, distribution='ones'):
    """
    Returns df of costs for each IP for each geo in geos_df according to distribution.

    Costs always sum to #IPS (i.e., len(IP_COLUMNS)).

    Available distributions:
        - 'ones': cost is 1 for each IP.
        - 'uniform': costs are sampled uniformly across IPs independently
                     for each geo.
    """
    # Copy the countries and regions dataset in order to add IP columns
    df = geos_df.copy()

    assert distribution in ['ones', 'uniform'], \
           f'Unsupported distribution {distribution}'

    if distribution == 'ones':
        df[IP_COLUMNS] = 1

    elif distribution == 'uniform':

        # Generate weights uniformly for each geo independently.
        nb_geos = len(df)
        nb_ips = len(IP_COLUMNS)
        samples = np.random.uniform(size=(nb_ips, nb_geos))
        weights = nb_ips * samples / samples.sum(axis=0)
        df[IP_COLUMNS] = weights.T

        # Round weights for better readability with negligible loss of generality.
        df = df.round(2)

    return df


def load_geos(path_to_geo_file):
    print(f"Loading countries and regions from {path_to_geo_file}")
    geos_df = pd.read_csv(path_to_geo_file,
                          encoding="ISO-8859-1",
                          dtype={"RegionName": str})
    return geos_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution",
                        type=str,
                        required=True,
                        help="Distribution to generate weights from. Current"
                              "options are 'ones', and 'uniform'.")
    parser.add_argument("-c", "--countries_path",
                        dest="countries_path",
                        type=str,
                        required=False,
                        default=DEFAULT_GEOS,
                        help="The path to a csv file containing the list of countries and regions to use. "
                             "The csv file must contain the following columns: CountryName,RegionName "
                             "and names must match latest Oxford's ones")
    parser.add_argument("-o", "--output_file",
                        type=str,
                        required=True,
                        help="Name of csv file to write generated weights to.")
    args = parser.parse_args()

    print(f"Generating weights with distribution {args.distribution}...")
    weights_df = generate_costs_for_geos_file(args.countries_path, args.distribution)
    print("Writing weights to file...")
    weights_df.to_csv(args.output_file, index=False)
    print("Done. Thank you.")
