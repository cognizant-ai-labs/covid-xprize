"""
Reads in pre-generated predictions from S3 bucket `S3_BUCKET` for each predictor
Ranks the predictions and generates rankings. Uploads ranking to S3 for the UI to display, and whatever else
"""

import logging.config
import os
from datetime import date

import numpy as np
import pandas as pd
import s3fs
from pandas import DataFrame

from common.args import parse_args
from common.constants import Constants
from common.common_routines import load_dataset
from validation.validation import validate_submission

# Set up logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('robojudge')

NUM_PREV_DAYS_TO_INCLUDE = 6
WINDOW_SIZE = 7

# Wrapper object for accessing S3
FS = s3fs.S3FileSystem()


def get_actual_cases(df, start_date, end_date):
    # 1 day earlier to compute the daily diff
    start_date_for_diff = start_date - pd.offsets.Day(WINDOW_SIZE)
    actual_df = df[["CountryName", "RegionName", "Date", "ConfirmedCases"]]
    # Filter out the data set to include all the data needed to compute the diff
    actual_df = actual_df[(actual_df.Date >= start_date_for_diff) & (actual_df.Date <= end_date)]
    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    actual_df["GeoID"] = np.where(actual_df["RegionName"].isnull(),
                                  actual_df["CountryName"],
                                  actual_df["CountryName"] + ' / ' + actual_df["RegionName"])
    actual_df.sort_values(by=["GeoID", "Date"], inplace=True)
    # Compute the diff
    actual_df["ActualDailyNewCases"] = actual_df.groupby("GeoID")["ConfirmedCases"].diff().fillna(0)
    # Compute the 7 day moving average
    actual_df["ActualDailyNewCases7DMA"] = actual_df.groupby(
        "GeoID")['ActualDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)
    return actual_df


def get_predictions_from_file(predictor_name, predictions_file, ma_df):
    preds_df = pd.read_csv(predictions_file,
                           parse_dates=['Date'],
                           encoding="ISO-8859-1",
                           error_bad_lines=False)
    preds_df["RegionName"] = preds_df["RegionName"].fillna("")
    preds_df["PredictorName"] = predictor_name
    preds_df["Prediction"] = True

    # Append the true number of cases before start date
    ma_df["PredictorName"] = predictor_name
    ma_df["Prediction"] = False
    preds_df = ma_df.append(preds_df, ignore_index=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    preds_df["GeoID"] = np.where(preds_df["RegionName"].isnull(),
                                 preds_df["CountryName"],
                                 preds_df["CountryName"] + ' / ' + preds_df["RegionName"])
    # Sort
    #     preds_df.sort_values(by=["CountryName","RegionName", "Date"], inplace=True)
    preds_df.sort_values(by=["GeoID", "Date"], inplace=True)
    # Compute the 7 days moving average for PredictedDailyNewCases
    preds_df["PredictedDailyNewCases7DMA"] = preds_df.groupby(
        "GeoID")['PredictedDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)

    # Put PredictorName first
    preds_df = preds_df[["PredictorName"] + [col for col in preds_df.columns if col != "PredictorName"]]
    return preds_df


def rank_submissions(start_date: str, end_date: str, submissions_date: str) -> DataFrame:
    actual_df, ma_df = _get_reference_datasets(start_date, end_date)

    # Find submissions in S3
    teams_folder = f's3://{Constants.S3_BUCKET}/predictions/{submissions_date}/teams'
    teams = FS.ls(teams_folder)
    prediction_file_name = start_date + "_" + end_date + ".csv"
    ranking_df = pd.DataFrame()

    for team in teams:
        # Get just team name without full path
        team_name = team.rsplit('/', 1)[1]

        predictions_file_path = f'{teams_folder}/{team_name}/{prediction_file_name}'

        if not FS.exists(predictions_file_path):
            LOGGER.warning(f'Predictions file not found for team "{team_name}": {predictions_file_path}. '
                           f'Cannot rank this team.')
            continue

        LOGGER.info(f'Ranking submission for team: "{team_name}"')
        # validate
        errors = validate_submission(start_date, end_date, predictions_file_path)
        if not errors:
            LOGGER.info(f'"{team_name}" submission passes validation')
            preds_df = get_predictions_from_file(team_name, predictions_file_path, ma_df)
            merged_df = actual_df.merge(preds_df, on=['CountryName', 'RegionName', 'Date', 'GeoID'], how='left')
            ranking_df = ranking_df.append(merged_df)
        else:
            LOGGER.warning(f'Team "{team_name}" did not submit valid predictions! Errors: ')
            LOGGER.warning('\n'.join(errors))

    ranking_df['DiffDaily'] = (ranking_df["ActualDailyNewCases"] - ranking_df["PredictedDailyNewCases"]).abs()
    ranking_df['Diff7DMA'] = (ranking_df["ActualDailyNewCases7DMA"] - ranking_df["PredictedDailyNewCases7DMA"]).abs()

    # Compute the cumulative sum of 7DMA errors
    ranking_df['CumulDiff7DMA'] = ranking_df.groupby(["GeoID", "PredictorName"])['Diff7DMA'].cumsum()

    # Keep only predictions (either Prediction == True) or on or after start_date
    ranking_df = ranking_df[ranking_df["Date"] >= start_date]

    # Sort by 7 days moving average diff
    ranking_df.sort_values(by=["CountryName", "RegionName", "Date", "Diff7DMA"], inplace=True)

    ranking_df.groupby('PredictorName').Diff7DMA.sum().sort_values()

    ranking_df[(ranking_df.CountryName == "United States") & (ranking_df.RegionName == "")].groupby(
        ["PredictorName"]).Diff7DMA.sum().sort_values()

    return ranking_df


def _get_reference_datasets(start_date_str: str, end_date_str: str) -> (DataFrame, DataFrame):
    # Convert to real datetime objects
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    latest_df = load_dataset()
    actual_df = get_actual_cases(latest_df, start_date, end_date)
    ma_df = actual_df[actual_df["Date"] < start_date]
    ma_df = ma_df[["CountryName", "RegionName", "Date", "ActualDailyNewCases"]]
    ma_df = ma_df.rename(columns={"ActualDailyNewCases": "PredictedDailyNewCases"})
    return actual_df, ma_df


def upload_to_s3(today_date, ranking_df):
    s3_destination = f's3://{Constants.S3_BUCKET}/predictions/{today_date}/rankings/ranking.csv'
    LOGGER.info(f'Uploading rankings to S3: {s3_destination}')
    ranking_df.to_csv(s3_destination, index=False)


if __name__ == '__main__':
    args = parse_args(with_ip=False)

    today_date = date.today().strftime("%Y_%m_%d")
    LOGGER.info(f'Generating rankings for {today_date} start date {args.start_date} end date {args.end_date}...')
    rankings = rank_submissions(args.start_date, args.end_date, today_date)
    upload_to_s3(today_date, rankings)
    LOGGER.info(f'Done with rankings for for {today_date} start date {args.start_date} end date {args.end_date}')

