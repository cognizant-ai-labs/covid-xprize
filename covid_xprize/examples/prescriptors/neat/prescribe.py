# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy

import neat

# Path to file containing neat prescriptors. Here we simply use a
# recent checkpoint of the population from train_prescriptor.py,
# but this is likely not the most complementary set of prescriptors.
# Many approaches can be taken to generate/collect more diverse sets.
# Note: this set can contain up to 10 prescriptors for evaluation.
from covid_xprize.examples.prescriptors.neat.utils import prepare_historical_df, CASES_COL, IP_COLS, IP_MAX_VALUES, \
    add_geo_id, get_predictions, PRED_CASES_COL

PRESCRIPTORS_FILE = 'neat-checkpoint-0'

# Number of days the prescriptors look at in the past.
NB_LOOKBACK_DAYS = 14


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Load historical data with basic preprocessing
    print("Loading historical data...")
    df = prepare_historical_df()

    # Restrict it to dates before the start_date
    df = df[df['Date'] <= start_date]

    # Fill in any missing case data using predictor given ips_df.
    # todo: ignore ips_df for now, and instead assume we have case
    # data for all days and geos up until the start_date.

    # Create historical data arrays for all geos
    past_cases = {}
    past_ips = {}
    for geo in df['GeoID'].unique():
        geo_df = df[df['GeoID'] == geo]
        past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))
        past_ips[geo] = np.array(geo_df[IP_COLS])

    # Gather values for scaling network output
    ip_max_values_arr = np.array([IP_MAX_VALUES[ip] for ip in IP_COLS])

    # Load prescriptors
    checkpoint = neat.Checkpointer.restore_checkpoint(PRESCRIPTORS_FILE)
    prescriptors = checkpoint.population.values()
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-prescriptor')

    # Load IP costs to condition prescriptions
    cost_df = pd.read_csv(path_to_cost_file)
    cost_df['RegionName'] = cost_df['RegionName'].fillna("")
    cost_df = add_geo_id(cost_df)
    geo_costs = {}
    for geo in cost_df['GeoID'].unique():
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

        # Generate prescriptions one day at a time, feeding resulting
        # predictions from the predictor back into the prescriptor.
        for date in pd.date_range(start_date, end_date):
            date_str = date.strftime("%Y-%m-%d")

            # Get prescription for all regions
            for geo in df['GeoID'].unique():

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

                # Add it to prescription dictionary
                country_name, region_name = geo.split('__')
                if region_name == 'nan':
                    region_name = np.nan
                df_dict['CountryName'].append(country_name)
                df_dict['RegionName'].append(region_name)
                df_dict['Date'].append(date_str)
                for ip_col, prescribed_ip in zip(IP_COLS, prescribed_ips):
                    df_dict[ip_col].append(prescribed_ip)

            # Create dataframe from prescriptions
            pres_df = pd.DataFrame(df_dict)

            # Make prediction given prescription for all countries
            pred_df = get_predictions(start_date_str, date_str, pres_df)

            # Update past data with new day of prescriptions and predictions
            pres_df['GeoID'] = pres_df['CountryName'] + '__' + pres_df['RegionName'].astype(str)
            pred_df['RegionName'] = pred_df['RegionName'].fillna("")
            pred_df['GeoID'] = pred_df['CountryName'] + '__' + pred_df['RegionName'].astype(str)
            new_pres_df = pres_df[pres_df['Date'] == date_str]
            new_pred_df = pred_df[pred_df['Date'] == date_str]
            for geo in df['GeoID'].unique():
                geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]

                # Append array of prescriptions
                pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in IP_COLS]).reshape(1,-1)
                eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

                # It is possible that the predictor does not return values for some regions.
                # To make sure we generate full prescriptions, this script continues anyway.
                # Geos that are ignored in this way by the predictor, will not be used in
                # quantitative evaluation. A list of such geos can be found in unused_geos.txt.
                if len(geo_pred) != 0:
                    eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                     geo_pred[PRED_CASES_COL].values[0])

        # Add prescription df to list of all prescriptions for this submission
        pres_df['PrescriptionIndex'] = prescription_idx
        prescription_dfs.append(pres_df)

    # Combine dfs for all prescriptions into a single df for the submission
    prescription_df = pd.concat(prescription_dfs)

    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

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
