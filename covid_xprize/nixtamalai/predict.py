# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from microtc.utils import load_model
from covid_xprize.nixtamalai.helpers import preprocess_npi
from covid_xprize.nixtamalai.helpers import ID_COLS
from collections import defaultdict


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, "models", "kmeans.model")


def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path) -> None:
    """
    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    preds_df = predict_df(start_date, end_date, path_to_ips_file, verbose=False)
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")


def predict_df(start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
    """
    Generates a file with daily new cases predictions for the given countries, regions and npis, between
    start_date and end_date, included.
    :param start_date_str: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date_str: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception_date and end_date
    :param verbose: True to print debug logs
    :return: a Pandas DataFrame containing the predictions
    """
    # Load historical intervention plans, since inception
    hist_ips_df = pd.read_csv(path_to_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)
    hist_ips_df = preprocess_npi(hist_ips_df)
    trans, model = load_model(MODEL_FILE)
    output = defaultdict(list)
    for X in trans.transform(hist_ips_df, start_date_str, end_date_str):
        hy = trans.update_prediction(model.predict(X))
        key = X.iloc[0]["GeoID"]
        output[key].append(hy)
    geo_pred_dfs = list()
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')    
    hist_ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]
    for key, value in output.items():
        geo_pred_df = hist_ips_df.loc[hist_ips_df.GeoID == key, ID_COLS].copy()
        geo_pred_df['PredictedDailyNewCases'] = value
        geo_pred_dfs.append(geo_pred_df)
    pred_df = pd.concat(geo_pred_dfs)
    # Drop GeoID column to match expected output format
    pred_df = pred_df.drop(columns=['GeoID'])
    return pred_df


# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
