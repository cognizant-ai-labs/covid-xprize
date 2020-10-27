# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os
import pickle

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, "models", "model.pkl")
DATA_FILE = os.path.join(ROOT_DIR, 'data', "OxCGRT_latest.csv")
ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COL = ['NewCases']
NPI_COLS = ['C1_School closing',
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
NB_LOOKBACK_DAYS = 30
# For testing, restrict training data to that before a hypothetical predictor submission date
HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-07-31")


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
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Load historical intervention plans, since inception
    hist_ips_df = pd.read_csv(path_to_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data",
    hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)
    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in NPI_COLS:
        hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    # Intervention plans to forecast for: those between start_date and end_date
    ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]

    # Load historical data to use in making predictions in the same way
    # This is the data we trained on
    # We stored it locally as for predictions there will be no access to the internet
    hist_cases_df = pd.read_csv(DATA_FILE,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    hist_cases_df['GeoID'] = hist_cases_df['CountryName'] + '__' + hist_cases_df['RegionName'].astype(str)
    # Add new cases column
    hist_cases_df['NewCases'] = hist_cases_df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    # Fill any missing case values by interpolation and setting NaNs to 0
    hist_cases_df.update(hist_cases_df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))
    # Keep only the id and cases columns
    hist_cases_df = hist_cases_df[ID_COLS + CASES_COL]

    # Load model
    with open(MODEL_FILE, 'rb') as model_file:
        model = pickle.load(model_file)

    # Make predictions for each country,region pair
    geo_pred_dfs = []
    for g in ips_df.GeoID.unique():
        if verbose:
            print('\nPredicting for', g)

        # Pull out all relevant data for country c
        hist_cases_gdf = hist_cases_df[hist_cases_df.GeoID == g]
        last_known_date = hist_cases_gdf.Date.max()
        ips_gdf = ips_df[ips_df.GeoID == g]
        past_cases = np.array(hist_cases_gdf[CASES_COL])
        past_npis = np.array(hist_ips_df[NPI_COLS])
        future_npis = np.array(ips_gdf[NPI_COLS])

        # Make prediction for each day
        geo_preds = []
        # Start predicting from start_date, unless there's a gap since last known date
        current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
        days_ahead = 0
        while current_date <= end_date:
            # Prepare data
            X_cases = past_cases[-NB_LOOKBACK_DAYS:]
            X_npis = past_npis[-NB_LOOKBACK_DAYS:]
            X = np.concatenate([X_cases.flatten(),
                                X_npis.flatten()])

            # Make the prediction (reshape so that sklearn is happy)
            pred = model.predict(X.reshape(1, -1))[0]
            pred = max(0, pred)  # Do not allow predicting negative cases
            # Add if it's a requested date
            if current_date >= start_date:
                geo_preds.append(pred)
                if verbose:
                    print(f"{current_date.strftime('%Y-%m-%d')}: {pred}")
            else:
                if verbose:
                    print(f"{current_date.strftime('%Y-%m-%d')}: {pred} - Skipped (intermediate missing daily cases)")

            # Append the prediction and npi's for next day
            # in order to rollout predictions for further days.
            past_cases = np.append(past_cases, pred)
            past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + 1], axis=0)

            # Move to next day
            current_date = current_date + np.timedelta64(1, 'D')
            days_ahead += 1

        # Create geo_pred_df with pred column
        geo_pred_df = ips_gdf[ID_COLS].copy()
        geo_pred_df['PredictedDailyNewCases'] = geo_preds
        geo_pred_dfs.append(geo_pred_df)

    # Combine all predictions into a single dataframe
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
