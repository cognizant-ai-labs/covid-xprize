# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy
from datetime import datetime

import neat

# Function imports from utils
from pathlib import Path

from covid_xprize.examples.prescriptors.neat.utils import add_geo_id
from covid_xprize.examples.prescriptors.neat.utils import get_predictions
from covid_xprize.examples.prescriptors.neat.utils import load_ips_file
from covid_xprize.examples.prescriptors.neat.utils import prepare_historical_df

# Constant imports from utils
from covid_xprize.examples.prescriptors.neat.utils import CASES_COL
from covid_xprize.examples.prescriptors.neat.utils import IP_COLS
from covid_xprize.examples.prescriptors.neat.utils import IP_MAX_VALUES
from covid_xprize.examples.prescriptors.neat.utils import PRED_CASES_COL

# Path to where this script lives
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Config file for running NEAT (expected to reside in same dir as this script)
NEAT_CONFIG_FILE = ROOT_DIR / 'config-prescriptor'

# Path to file containing neat prescriptors. Here we simply use a
# recent checkpoint of the population from train_prescriptor.py,
# but this is likely not the most complementary set of prescriptors.
# Many approaches can be taken to generate/collect more diverse sets.
# Note: this set can contain up to 10 prescriptors for evaluation.
PRESCRIPTORS_FILE = 'neat-checkpoint-0'

# Number of days the prescriptors look at in the past.
NB_LOOKBACK_DAYS = 14

# Number of prescriptions to make per country.
# This can be set based on how many solutions in PRESCRIPTORS_FILE
# we want to run and on time constraints.
NB_PRESCRIPTIONS = 3

# Number of days to fix prescribed IPs before changing them.
# This could be a useful toggle for decision makers, who may not
# want to change policy every day. Increasing this value also
# can speed up the prescriptor, at the cost of potentially less
# interesting prescriptions.
ACTION_DURATION = 15


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Load the past IPs data
    print("Loading past IPs data...")
    past_ips_df = load_ips_file(path_to_prior_ips_file)
    geos = past_ips_df['GeoID'].unique()

    # Load historical data with basic preprocessing
    print("Loading historical data...")
    df = prepare_historical_df()

    # Restrict it to dates before the start_date
    df = df[df['Date'] <= start_date]

    # Create past case data arrays for all geos
    past_cases = {}
    for geo in geos:
        geo_df = df[df['GeoID'] == geo]
        past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))

    # Create past ip data arrays for all geos
    past_ips = {}
    for geo in geos:
        geo_df = past_ips_df[past_ips_df['GeoID'] == geo]
        past_ips[geo] = np.array(geo_df[IP_COLS])

    # Fill in any missing case data before start_date
    # using predictor given past_ips_df.
    # Note that the following assumes that the df returned by prepare_historical_df()
    # has the same final date for all regions. This has been true so far, but relies
    # on it being true for the Oxford data csv loaded by prepare_historical_df().
    last_historical_data_date_str = df['Date'].max()
    last_historical_data_date = pd.to_datetime(last_historical_data_date_str,
                                               format='%Y-%m-%d')
    if last_historical_data_date + pd.Timedelta(days=1) < start_date:
        print("Filling in missing data...")
        missing_data_start_date = last_historical_data_date + pd.Timedelta(days=1)
        missing_data_start_date_str = datetime.strftime(missing_data_start_date,
                                                           format='%Y-%m-%d')
        missing_data_end_date = start_date - pd.Timedelta(days=1)
        missing_data_end_date_str = datetime.strftime(missing_data_end_date,
                                                           format='%Y-%m-%d')
        pred_df = get_predictions(missing_data_start_date_str,
                                  missing_data_end_date_str,
                                  past_ips_df)
        pred_df = add_geo_id(pred_df)
        for geo in geos:
            geo_df = pred_df[pred_df['GeoID'] == geo].sort_values(by='Date')
            pred_cases_arr = np.array(geo_df[PRED_CASES_COL])
            past_cases[geo] = np.append(past_cases[geo], pred_cases_arr)
    else:
        print("No missing data.")

    # Gather values for scaling network output
    ip_max_values_arr = np.array([IP_MAX_VALUES[ip] for ip in IP_COLS])

    # Load prescriptors
    checkpoint = neat.Checkpointer.restore_checkpoint(PRESCRIPTORS_FILE)
    prescriptors = list(checkpoint.population.values())[:NB_PRESCRIPTIONS]
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         NEAT_CONFIG_FILE)

    # Load IP costs to condition prescriptions
    cost_df = pd.read_csv(path_to_cost_file)
    cost_df['RegionName'] = cost_df['RegionName'].fillna("")
    cost_df = add_geo_id(cost_df)
    geo_costs = {}
    for geo in geos:
        costs = cost_df[cost_df['GeoID'] == geo]
        cost_arr = np.array(costs[IP_COLS])[0]
        geo_costs[geo] = cost_arr

    # Generate prescriptions
    prescription_dfs = []
    for prescription_idx, prescriptor in enumerate(prescriptors):
        print("Generating prescription", prescription_idx, "...")

        # Create net from genome
        net = neat.nn.FeedForwardNetwork.create(prescriptor, config)

        # Set up dictionary for keeping track of prescription
        df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
        for ip_col in sorted(IP_MAX_VALUES.keys()):
            df_dict[ip_col] = []

        # Set initial data
        eval_past_cases = deepcopy(past_cases)
        eval_past_ips = deepcopy(past_ips)

        # Generate prescriptions iteratively, feeding resulting
        # predictions from the predictor back into the prescriptor.
        action_start_date = start_date
        while action_start_date <= end_date:

            # Get prescription for all regions
            for geo in geos:

                # Prepare input data. Here we use log to place cases
                # on a reasonable scale; many other approaches are possible.
                X_cases = np.log(eval_past_cases[geo][-NB_LOOKBACK_DAYS:] + 1)
                X_ips = eval_past_ips[geo][-NB_LOOKBACK_DAYS:]
                X_costs = geo_costs[geo]
                X = np.concatenate([X_cases.flatten(),
                                    X_ips.flatten(),
                                    X_costs])

                # Get prescription
                prescribed_ips = net.activate(X)

                # Map prescription to integer outputs
                prescribed_ips = (prescribed_ips * ip_max_values_arr).round()

                # Add it to prescription dictionary for the full ACTION_DURATION
                country_name, region_name = geo.split('__')
                if region_name == 'nan':
                    region_name = np.nan
                for date in pd.date_range(action_start_date, periods=ACTION_DURATION):
                    if date > end_date:
                        break
                    date_str = date.strftime("%Y-%m-%d")
                    df_dict['CountryName'].append(country_name)
                    df_dict['RegionName'].append(region_name)
                    df_dict['Date'].append(date_str)
                    for ip_col, prescribed_ip in zip(IP_COLS, prescribed_ips):
                        df_dict[ip_col].append(prescribed_ip)

            # Create dataframe from prescriptions
            pres_df = pd.DataFrame(df_dict)

            # Make prediction given prescription for all countries
            pred_df = get_predictions(start_date_str, date_str, pres_df)

            # Update past data with new days of prescriptions and predictions
            pres_df = add_geo_id(pres_df)
            pred_df = add_geo_id(pred_df)
            for date in pd.date_range(action_start_date, periods=ACTION_DURATION):
                if date > end_date:
                    break
                date_str = date.strftime("%Y-%m-%d")
                new_pres_df = pres_df[pres_df['Date'] == date_str]
                new_pred_df = pred_df[pred_df['Date'] == date_str]
                for geo in geos:
                    geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                    geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                    # Append array of prescriptions
                    pres_arr = np.array([geo_pres[ip_col].values[0] for
                                         ip_col in IP_COLS]).reshape(1,-1)
                    eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

                    # It is possible that the predictor does not return values for some regions.
                    # To make sure we generate full prescriptions, this script continues anyway.
                    # This should not happen, but is included here for robustness.
                    if len(geo_pred) != 0:
                        eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                         geo_pred[PRED_CASES_COL].values[0])

            # Move on to next action date
            action_start_date += pd.DateOffset(days=ACTION_DURATION)

        # Add prescription df to list of all prescriptions for this submission
        pres_df['PrescriptionIndex'] = prescription_idx
        prescription_dfs.append(pres_df)

    # Combine dfs for all prescriptions into a single df for the submission
    prescription_df = pd.concat(prescription_dfs)
    prescription_df = prescription_df.drop(columns='GeoID')

    # Create the output directory if necessary.
    output_dir = os.path.dirname(output_file_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    # Save to a csv file
    prescription_df.to_csv(output_file_path, index=False)
    print('Prescriptions saved to', output_file_path)

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
                        dest="prior_ips_file",
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
    prescribe(args.start_date, args.end_date, args.prior_ips_file, args.cost_file, args.output_file)
    print("Done!")
