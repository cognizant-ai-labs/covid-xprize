# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.
import argparse
import logging.config
import os
import urllib.request

import numpy as np
import pandas as pd

from covid_xprize.scoring.predictor_scoring import load_dataset

logging.basicConfig(
    format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger('scenario_generator')

# See https://github.com/OxCGRT/covid-policy-tracker
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
ID_COLS = ['CountryName',
           'RegionName',
           'Date']
NPI_COLUMNS = ['C1_School closing',
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
# From https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md
MIN_NPIS = [0] * len(NPI_COLUMNS)
MAX_NPIS = [3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4]  # Sum is 34
INCEPTION_DATE = pd.to_datetime("2020-01-01", format='%Y-%m-%d')


def get_raw_data(cache_file, latest=True):
    """
    Returns the raw data from which to generate scenarios.
    Args:
        cache_file: the file to use to cache the data
        latest: True to force a download of the latest data and update cache_file,
                False to get the data from cache_file

    Returns: a Pandas DataFrame

    """
    # Download and cache the raw data file if it doesn't exist
    if not os.path.exists(cache_file) or latest:
        urllib.request.urlretrieve(DATA_URL, cache_file)
    latest_df = pd.read_csv(cache_file,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    # Fill any missing NPIs by assuming they are the same as previous day, or 0 if none is available
    latest_df.update(latest_df.groupby(['CountryName', 'RegionName'])[NPI_COLUMNS].ffill().fillna(0))
    return latest_df


def generate_scenario(start_date_str, end_date_str, raw_df, countries=None, scenario="Freeze"):
    """
    Generates a scenario: a list of intervention plans, with history since 1/1/2020.
    Args:
        start_date_str: start_date from which to apply the scenario. None to apply from last known date
        end_date_str: end_date of the data
        raw_df: the original data frame containing the raw data
        countries: a list of CountryName, or None for all countries
        scenario:
            - "Historical" to keep historical NPIs.
            - "Freeze" to apply the last available NPIs to every day between start_date and end_date, included
            - "MIN" to set all NPIs to 0 (i.e. plan is to take no measures)
            - "MAX" to set all NPIs to maximum values (i.e. plan is to do everything possible)
            - an array of size "number of days between start_date and end_date"
            containing for each day the array of integers of NPI_COLUMNS lengths to use.
        In case NPIs are not know BEFORE start_date, the last known ones are carried over.

    Returns: a Pandas DataFrame

    """
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    if start_date:
        if end_date < start_date:
            raise ValueError(f"end_date {end_date} cannot be before start_date {start_date}")

        if start_date < INCEPTION_DATE:
            raise ValueError(f"start_date {start_date} must be on or after inception date {INCEPTION_DATE}")

    ips_df = raw_df[ID_COLS + NPI_COLUMNS].copy()

    # Filter on countries
    if countries:
        ips_df = ips_df[ips_df.CountryName.isin(countries)]

    # Fill any missing "supposedly known" NPIs by assuming they are the same as previous day, or 0 if none is available
    for npi_col in NPI_COLUMNS:
        ips_df.update(ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    if scenario == "Historical":
        return ips_df

    for country in ips_df.CountryName.unique():
        all_regions = ips_df[ips_df.CountryName == country].RegionName.unique()
        for region in all_regions:
            new_rows = []
            ips_gdf = ips_df[(ips_df.CountryName == country) &
                             (ips_df.RegionName == region)]
            country_name = ips_gdf.iloc[0].CountryName
            region_name = ips_gdf.iloc[0].RegionName
            last_known_date = ips_gdf.Date.max()
            # If the start date is not specified, start from the day after the last known date
            if not start_date_str:
                start_date = last_known_date + np.timedelta64(1, 'D')
            # If the last known date is BEFORE the start date, start applying the scenario at last_known date
            current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
            scenario_to_apply = 0
            while current_date <= end_date:
                new_row = [country_name, region_name, current_date]
                if current_date < start_date:
                    # We're before the scenario start date. Carry over last known NPIs
                    npis = list(ips_gdf[ips_gdf.Date == last_known_date][NPI_COLUMNS].values[0])
                else:
                    # We are between start_date and end_date: apply the scenario
                    if scenario == "MIN":
                        npis = MIN_NPIS
                    elif scenario == "MAX":
                        npis = MAX_NPIS
                    elif scenario == "Freeze":
                        if start_date <= last_known_date:
                            day_before_start = max(INCEPTION_DATE, start_date - np.timedelta64(1, 'D'))
                            npis = list(ips_gdf[ips_gdf.Date == day_before_start][NPI_COLUMNS].values[0])
                        else:
                            npis = list(ips_gdf[ips_gdf.Date == last_known_date][NPI_COLUMNS].values[0])
                    else:
                        npis = scenario[scenario_to_apply]
                        scenario_to_apply = scenario_to_apply + 1
                new_row = new_row + npis
                new_rows.append(new_row)
                # Move to next day
                current_date = current_date + np.timedelta64(1, 'D')
            # Add the new rows
            if new_rows:
                new_rows_df = pd.DataFrame(new_rows, columns=ips_df.columns)
                # Delete any old row that might have been replaced by a scenario one for this country / region
                replaced_dates = list(new_rows_df["Date"].unique())
                rows_to_drop = ips_df[(ips_df.CountryName == country) &
                                      (ips_df.RegionName == region) &
                                      (ips_df.Date.isin(replaced_dates)) == True]
                ips_df.drop(rows_to_drop.index, axis=0, inplace=True)
                # Append the new rows
                ips_df = ips_df.append(new_rows_df)
                # Sort
                ips_df.sort_values(by=ID_COLS, inplace=True)

    return ips_df


def phase1_update(latest_df):
    # Feb 2, 2021: Handle US Virgin Islands: was a region of 'United States' for phase 1, but is now a country
    latest_df.loc[latest_df.CountryName == "United States Virgin Islands", "RegionName"] = "Virgin Islands"
    latest_df.loc[latest_df.CountryName == "United States Virgin Islands", "CountryName"] = "United States"
    return latest_df


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
    parser.add_argument("-c", "--countries_path",
                        dest="countries_path",
                        type=str,
                        required=False,
                        help="The path to a csv file containing the list of countries and regions to use. "
                             "The csv file must contain the following columns: CountryName,RegionName "
                             "and names must match latest Oxford's ones")
    parser.add_argument("-o", "--output_path",
                        dest="output_path",
                        type=str,
                        required=True,
                        help="The path to where the generated scenario CSV file should be written "
                             "including the filename. For example: /tmp/my_scenario.csv")
    parser.add_argument('-p1', '--phase1',
                        dest='phase1',
                        help="True to make the generated scenario backward compatible with Phase 1",
                        default=False, action='store_true')
    args = parser.parse_args()
    LOGGER.info("Generating scenario...")
    # Load the latest dataset from Oxford
    if args.countries_path:
        # Use the specified list of countries and regions
        latest_df = load_dataset(geos_file=args.countries_path)
    else:
        # Use the official list of countries and regions
        latest_df = load_dataset()

    # Fix the DataFrame to make it backward_compatible with phase 1's list of countries and regions
    if args.phase1:
        LOGGER.info("Making dataset backward compatible with Phase 1...")
        latest_df = phase1_update(latest_df)

    LOGGER.info("Dataset loaded.")
    LOGGER.info(f"Start date: {args.start_date}")
    LOGGER.info(f"End date: {args.end_date}")
    LOGGER.info("Generating...")
    scenario_df = generate_scenario(args.start_date,
                                    args.end_date,
                                    latest_df,
                                    countries=None,
                                    scenario="Freeze")
    LOGGER.info("Scenario created.")
    nb_countries = len(scenario_df.CountryName.unique())
    nb_regions = len(scenario_df.RegionName.unique()) - 1  # Ignore the 'nan' region
    nb_rows = len(scenario_df)
    nb_days = nb_rows / (nb_countries + nb_regions)
    LOGGER.info(f"{nb_countries} countries")
    LOGGER.info(f"{nb_regions} regions")
    LOGGER.info(f"{nb_rows} rows in generated file, which corresponds to {nb_days} days")
    # Save
    output_path = args.output_path
    LOGGER.info(f"Saving to: {output_path}")
    scenario_df.to_csv(output_path, index=False)
    LOGGER.info(f"Done!")


if __name__ == '__main__':
    do_main()
