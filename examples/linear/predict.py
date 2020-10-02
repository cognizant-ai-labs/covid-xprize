import argparse
import os

import pandas as pd

from examples.lstm.xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, "models.pkl")
DATA_FILE = os.path.join(ROOT_DIR, 'data', "OxCGRT_latest.csv")


def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str) -> None:
    """
    Generates a file with daily new cases predictions for the given countries, regions and intervention plans, between
    start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between start_date and end_date
    :return: Nothing. Saves a .csv file called 'start_date_end_date.csv'
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    # Save to a csv file
    output_file = start_date + "_" + end_date + ".csv"
    preds_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")


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
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file)
    print("Done!")
