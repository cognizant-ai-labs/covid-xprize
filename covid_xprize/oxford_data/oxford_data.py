# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

"""This module is a common interface for access to the Oxford online dataset and related processing routines and data structures.

* Populate data with country metadata - Population column

* Trim data to a date range

* Data smoothing with a window

* Generating window training samples"""

import os
from datetime import datetime
import pytz

import numpy as np
import pandas as pd
import requests
from pathlib import Path


# A link to the Oxford data set.
OXFORD_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker-legacy/main/legacy_data_202207/OxCGRT_latest.csv'
OXFORD_DATA_CACHE_FILE_PATH = Path(__file__).parent / 'data' / 'OxCGRT_latest.csv'

# Paths to population metadata.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = os.path.join(DATA_PATH, "brazil_populations.csv")
US_PREFIX = "United States / "


# Names of key columns in the Oxford data.
CONTEXT_COLUMNS = [
    'GeoID',
    'CountryCode',
    'Date',
    'ConfirmedCases',
    'ConfirmedDeaths',
    'Population'
]
ID_COLS = [
    'CountryName',
    'RegionName',
    'Date'
]
NPI_COLUMNS = [
    'C1_School closing',
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
    'H6_Facial Coverings'
]


# Parameters for preprocessing.
MIN_CASES = 10
WINDOW_SIZE = 7


def _load_geospatial_df(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        dtype={"RegionName": str,
                                "RegionCode": str},
                        on_bad_lines='skip',
                        low_memory=False)


def load_original_oxford_data() -> pd.DataFrame:
    """Loads the original Oxford dataset with no preprocessing."""
    if not OXFORD_DATA_CACHE_FILE_PATH.exists():
        data = requests.get(OXFORD_DATA_URL)
        with open(OXFORD_DATA_CACHE_FILE_PATH, 'wb') as f:
            f.write(data.content)
    return _load_geospatial_df(OXFORD_DATA_CACHE_FILE_PATH)


def load_oxford_data_trimmed(end_date: str, start_date: str = None) -> pd.DataFrame:
    """
    Loads the very original Oxford dataset and removes dates after the given end_date.

    :param end_date: The final date of data to return, format YYYY-MM-DDD.
    """
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    df = load_original_oxford_data()
    df = df[(df.Date <= end_date)]
    if start_date is not None:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        df = df[(df.Date >= start_date)]
    return df


def _add_geoid_column(df: pd.DataFrame) -> None:
    """Add GeoID column that combines CountryName and RegionName for easier manipulation of data."""
    # GeoID is CountryName / RegionName
    # np.where usage: if A then B else C
    df["GeoID"] = np.where( df["RegionName"].isnull(),
                            df["CountryName"],
                            df["CountryName"] + ' / ' + df["RegionName"])
    return df


def load_ips_file(file_path: str) -> pd.DataFrame:
    """
    Loads the intervention plans (IPs) in the given file. The IPs are assumed to have columns for the date and country.
    """
    df = _load_geospatial_df(file_path)
    _add_geoid_column(df)
    return df


def _fill_missing_values(df: pd.DataFrame) -> None:
    """
    Fill missing values by interpolation, ffill, and filling NaNs
    :param df: Dataframe to be filled
    """
    df.update(df.groupby('GeoID', group_keys=False).ConfirmedCases.apply(
        lambda group: group.interpolate(limit_area='inside')))
    # Drop country / regions for which no number of cases is available
    df.dropna(subset=['ConfirmedCases'], inplace=True)
    df.update(df.groupby('GeoID', group_keys=False).ConfirmedDeaths.apply(
        lambda group: group.interpolate(limit_area='inside')))
    # Drop country / regions for which no number of deaths is available
    df.dropna(subset=['ConfirmedDeaths'], inplace=True)
    for npi_column in NPI_COLUMNS:
        df.update(df.groupby('GeoID', group_keys=False)[npi_column].ffill().fillna(0))


def _load_additional_context_df() -> pd.DataFrame:
    # File containing the population for each country
    # Note: this file contains only countries population, not regions
    additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                        usecols=['CountryName', 'Population'])
    additional_context_df['GeoID'] = additional_context_df['CountryName']

    # US states population
    additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                            usecols=['NAME', 'POPESTIMATE2019'])
    # Rename the columns to match measures_df ones
    additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
    # Prefix with country name to match measures_df
    additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

    # UK population
    additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)

    # Brazil population
    additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)

    # Append the new data to additional_df
    additional_context_df = pd.concat([additional_context_df,
                                        additional_us_states_df,
                                        additional_uk_df,
                                        additional_brazil_df])

    return additional_context_df


def add_population_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add population column to df in order to compute performance per 100K of population.
    """
    pop_df = _load_additional_context_df()
    return df.merge(pop_df[['GeoID', 'Population']], on=['GeoID'], how='left', suffixes=('', '_y'))


def prepare_cases_dataframe(data_url: str = None) -> pd.DataFrame:
    """
    Loads the cases dataset from the given file, cleans it, and computes cases columns.
    :param data_url: the url containing the original data
    :return: a Pandas DataFrame with the historical data
    """
    if data_url is None:
        df = load_original_oxford_data()
        _add_geoid_column(df)
    else:
        df = load_ips_file(data_url)

    # Additional context df (e.g Population for each country)
    df = add_population_column(df)

    # Drop countries with no population data
    df.dropna(subset=['Population'], inplace=True)

    # Keep only needed columns
    columns = CONTEXT_COLUMNS + NPI_COLUMNS
    df = df[columns]

    # Fill in missing values
    _fill_missing_values(df)

    # Compute number of new cases and deaths each day
    df['NewCases'] = df.groupby('GeoID', group_keys=False).ConfirmedCases.diff().fillna(0)
    df['NewDeaths'] = df.groupby('GeoID', group_keys=False).ConfirmedDeaths.diff().fillna(0)

    # Replace negative values (which do not make sense for these columns) with 0
    df['NewCases'] = df['NewCases'].clip(lower=0)
    df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

    # Compute smoothed versions of new cases and deaths each day
    df['SmoothNewCases'] = df.groupby('GeoID', group_keys=False)['NewCases'].rolling(
        WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
    df['SmoothNewDeaths'] = df.groupby('GeoID', group_keys=False)['NewDeaths'].rolling(
        WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

    # Compute percent change in new cases and deaths each day
    df['CaseRatio'] = df.groupby('GeoID', group_keys=False).SmoothNewCases.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1
    df['DeathRatio'] = df.groupby('GeoID', group_keys=False).SmoothNewDeaths.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1

    # Add column for proportion of population infected
    df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

    # Create column of value to predict
    df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

    # Create column of for cases per 100K
    df['SmoothNewCasesPer100K'] = df['SmoothNewCases'] / (df['Population'] / 100_000)

    return df


def threshold_min_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all rows with too few cases"""
    return df.drop(df[df.ConfirmedCases < MIN_CASES].index)


def create_country_samples(df: pd.DataFrame,
                           countries: list[str],
                           context_column: str,
                           nb_test_days: int = 14,
                           nb_lookback_days: int = 21) -> dict[str, dict[str, np.ndarray]]:
    """
    For each country, creates numpy arrays for Keras
    :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
    :param countries: a list of country names
    :param context_column: the column of the df to use as context and outcome
    :return: a dictionary of train and test sets, for each specified country.
            The dictionary has the following keys:

                * `X_context`

                * `X_action`

                * `y`

                * `X_train_context`

                * `X_train_action`

                * `y_train`

                * `X_test_context`

                * `X_test_action`

                * `y_test`
    """
    action_columns = NPI_COLUMNS
    outcome_column = context_column
    country_samples = {}
    for c in countries:
        cdf = df[df.GeoID == c]
        cdf = cdf[cdf.ConfirmedCases.notnull()]
        context_data = np.array(cdf[context_column])
        action_data = np.array(cdf[action_columns])
        outcome_data = np.array(cdf[outcome_column])
        context_samples = []
        action_samples = []
        outcome_samples = []
        nb_total_days = outcome_data.shape[0]
        for d in range(nb_lookback_days, nb_total_days):
            context_samples.append(context_data[d - nb_lookback_days:d])
            action_samples.append(action_data[d - nb_lookback_days:d])
            outcome_samples.append(outcome_data[d])
        if len(outcome_samples) > 0:
            X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
            X_action = np.stack(action_samples, axis=0)
            y = np.stack(outcome_samples, axis=0)
            country_samples[c] = {
                'X_context': X_context,
                'X_action': X_action,
                'y': y,
                'X_train_context': X_context[:-nb_test_days],
                'X_train_action': X_action[:-nb_test_days],
                'y_train': y[:-nb_test_days],
                'X_test_context': X_context[-nb_test_days:],
                'X_test_action': X_action[-nb_test_days:],
                'y_test': y[-nb_test_days:],
            }
    return country_samples


def create_prediction_initial_context_and_action_vectors(
        df: pd.DataFrame,
        countries: list[str],
        context_column: str,
        start_date: datetime,
        nb_lookback_days: int = 21) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Creates the context and action vectors appropriate to use as inputs for the given prediction time.
    :param df: The dataframe with cases and NPIs.
    :param countries: The countries to get vectors for.
    :param contxt_column: The column of the df to use as context and outcome.
    :param start_date: Get context immediately before this start date.
    :return: A tuple (context_vectors, action_vectors), where each is a map from the GeoID to a numpy array.
    """
    context_vectors, action_vectors = {}, {}
    # With this date cutoff, the date corresponding to `start_date` will be the final `y` value,
    # And the final `X_test_context` vector will end immediately prior to `start_date`.
    df = df[df.Date <= start_date]
    country_samples = create_country_samples(df, countries, context_column, nb_lookback_days=nb_lookback_days)
    for c in countries:
        if c not in country_samples.keys():
            continue
        context_vectors[c] = country_samples[c]['X_test_context'][-1]
        action_vectors[c] =  country_samples[c]['X_test_action'][-1]
    return context_vectors, action_vectors


def convert_smooth_cases_per_100K_to_new_cases(smooth_cases_per_100K,
                                               window_size,
                                               prev_new_cases_list,
                                               pop_size):
    # Smooth cases per 100K -> Smooth cases
    smooth_cases = (pop_size / 100000.) * smooth_cases_per_100K
    # Solve:
    # (new_cases[-6:].sum() + x) / 7 = smooth_cases[0]
    # x = smooth_cases[0] * 7 - new_cases[-6:].sum()
    all_unsmooth_cases = np.zeros(smooth_cases_per_100K.shape[0] + window_size)
    all_unsmooth_cases[:window_size] = prev_new_cases_list[-window_size:]
    for i in range(window_size, all_unsmooth_cases.shape[0]):
        all_unsmooth_cases[i] = smooth_cases[i - window_size] * window_size - all_unsmooth_cases[i - (window_size-1): i].sum()
    unsmooth_cases = all_unsmooth_cases[window_size:]
    # Never return anything less than zero.
    return unsmooth_cases.clip(min=0.0)


def convert_prediction_ratios_to_new_cases(ratios: np.ndarray,
                                            window_size: int,
                                            prev_new_cases: np.ndarray,
                                            initial_total_cases: float,
                                            pop_size: float) -> np.ndarray:
    """Process a list of case ratios into a list of daily new cases.
    :param ratios: Shape (PredictionDays,)
        A column corresponding to the PredictionRatio.
    :param window_size: The number of days in the smoothing window.
    :param prev_new_cases: Shape (SmoothingWindow,). The array of NewCases leading up to the prediction window.
    :param initial_total_cases: The value of ConfirmedCases indicating the total cases on the first day of the prediction window.
    :param pop_size: The population of the region.
    :return: NewCases, an array corresponding to the input ratios.
    """
    new_new_cases = []
    prev_new_cases_list = list(prev_new_cases)
    curr_total_cases = initial_total_cases
    for ratio in ratios:
        new_cases = convert_ratio_to_new_cases(ratio,
                                                window_size,
                                                prev_new_cases_list,
                                                curr_total_cases / pop_size)
        # new_cases can't be negative!
        new_cases = max(0, new_cases)
        # Which means total cases can't go down
        curr_total_cases += new_cases
        # Update prev_new_cases_list for next iteration of the loop
        prev_new_cases_list.append(new_cases)
        new_new_cases.append(new_cases)
    return new_new_cases


def convert_ratio_to_new_cases(ratio: float,
                                window_size: int,
                                prev_new_cases_list: list[float],
                                prev_pct_infected: float) -> np.ndarray:
    """Convert the PredictionRatio column into the NewCases column.
    :param ratio: The value of PredictionRatio on the given day.
    :param window_size: The number of days in the smoothing window.
    :param prev_new_cases_list: The NewCases column values prior to, but not including, the given day.
    :param prev_pct_infected: The value of the ProportionInfected column on the given day. 
    :return: The value of the NewCases column on the given day."""
    # if len(prev_new_cases_list) < window_size:
    #     raise TypeError("System needs more samples for conversion.")
    # # Find the inverse transformation:
    # # PredictionRatio = CaseRatio / (1 - ProportionInfected)
    # # => CaseRatio = (1 - ProportionInfected) * PredictionRatio
    # case_ratio = ratio * (1 - prev_pct_infected)
    

    # # PredictionRatio = CaseRatio / (1 - ProportionInfected)
    # # => (1 - ProportionInfected) = CaseRatio/PredictionRatio
    # # => 1 - CaseRatio/PredictionRatio = ProportionInfected
    # # => CaseRatio = (1 - ProportionInfected) * PredictionRatio
    # case_ratio = ratio * (1 - prev_pct_infected)

    # # Find the inverse transformation:
    # # CaseRatio[1] = SmoothNewCases[1] / SmoothNewCases[0]
    # # => SmoothNewCases[1] = CaseRatio[1] * SmoothNewCases[0]
    # smooth_new_cases = case_ratio * np.mean(prev_new_cases_list[-window_size:])
    
    # # Find the inverse transformation:
    # # SmoothNewCases[7] = (1/7) * (NewCases[1] + ... + NewCases[7])
    # # => NewCases[7] = 7 * SmoothNewCases[7] - (NewCases[1] + ... + NewCases[6])
    # new_cases = window_size * smooth_new_cases - np.sum(prev_new_cases_list[-(window_size-1):])
    # return new_cases

    return (ratio * (1 - prev_pct_infected) - 1) * \
            (window_size * np.mean(prev_new_cases_list[-window_size:])) \
            + prev_new_cases_list[-window_size]


def _select_meta_cases_and_deaths_cols(df: pd.DataFrame) -> pd.DataFrame:
    desired_columns = ["CountryName", "RegionName", "Date",
                       "ConfirmedCases", "ConfirmedDeaths",
                       "PredictedDailyNewCases", "PredictedDailyNewDeaths"]
    take_columns = df.columns.intersection(desired_columns)
    return df[take_columns].copy(deep=True)


def _cut_to_range_and_add_diffs(
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        WINDOW_SIZE: int) -> pd.DataFrame:
    # 1 day earlier to compute the daily diff
    start_date_for_diff = start_date - pd.offsets.Day(WINDOW_SIZE)
    # Filter out the data set to include all the data needed to compute the diff
    df = df[(df.Date >= start_date_for_diff) & (df.Date <= end_date)].copy(deep=True)
    df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the diff
    for src_column, tgt_column in (
        ("ConfirmedCases", "ActualDailyNewCases"),
        ("ConfirmedDeaths", "ActualDailyNewDeaths"),
    ):
        if not src_column in df.columns:
            continue
        df[tgt_column] = df.groupby("GeoID", group_keys=False)[src_column].diff()
    return df


def _moving_average_inplace(df: pd.DataFrame, src_column: str, tgt_column: str, window_size: int):
    """In-place operation to add a moving average column.
    :param df: The Oxford dataframe.
    :param src_column: The name of the column to use as the data soruce.
    :param tgt_column: The name of the column to store the moving average.
    :param window_size: The moving average window size."""
    df[tgt_column] = df.groupby(
        "GeoID", group_keys=False)[src_column].rolling(
        window_size, center=False).mean().reset_index(0, drop=True)


def _cut_to_range_and_add_diffs_and_smooth_diffs(
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime) -> pd.DataFrame:
    """Compute moving average columns and add them to the dataframe.
    Works with both cases & deaths and both actual and predicted cases."""
    df = _cut_to_range_and_add_diffs(df, start_date, end_date, WINDOW_SIZE)
    for src_column, tgt_column in (
        ('ActualDailyNewCases', f'ActualDailyNewCases{WINDOW_SIZE}DMA'),
        ('ActualDailyNewDeaths', f'ActualDailyNewDeaths{WINDOW_SIZE}DMA'),
        ('PredictedDailyNewCases', f'PredictedDailyNewCases{WINDOW_SIZE}DMA'),
        ('PredictedDailyNewDeaths', f'PredictedDailyNewDeaths{WINDOW_SIZE}DMA'),
        ):
        if not src_column in df.columns:
            continue
        _moving_average_inplace(df, src_column, tgt_column, WINDOW_SIZE)
    return df


def _sort_by_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(by=ID_COLS, inplace=True, ignore_index=True)
    return df


def process_submission(dataset: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    df = dataset
    df = _select_meta_cases_and_deaths_cols(df)
    df = _add_geoid_column(df)
    df = _cut_to_range_and_add_diffs_and_smooth_diffs(df, start_date, end_date)
    df = _sort_by_id_cols(df)
    return df


def most_affected_countries(df: pd.DataFrame, nb_geos: int, min_historical_days: int) -> list[str]:
    """
    Returns the list of most affected countries, in terms of confirmed deaths.
    :param df: the data frame containing the historical data
    :param nb_geos: the number of geos to return
    :param min_historical_days: the minimum days of historical data the countries must have
    :return: a list of country names of size nb_countries if there were enough, and otherwise a list of all the
    country names that have at least min_look_back_days data points.
    """
    # Don't include the region-level data, just the country-level summaries.
    gdf = df[df.RegionName.isna()]
    gdf = gdf.groupby('CountryName', group_keys=False)['ConfirmedDeaths'].agg(['max', 'count']).sort_values(
        by='max', ascending=False)
    filtered_gdf = gdf[gdf["count"] > min_historical_days]
    geos = list(filtered_gdf.head(nb_geos).index)
    return geos


def most_affected_geos(df, nb_geos, min_historical_days):
    """
    Returns the list of most affected countries, in terms of confirmed deaths.
    :param df: the data frame containing the historical data
    :param nb_geos: the number of countries to return
    :param min_historical_days: the minimum days of historical data the countries must have
    :return: a list of country names of size nb_geos if there were enough, and otherwise a list of all the
    country names that have at least min_look_back_days data points.
    """
    # By default use most affected countries with enough history
    gdf = df.groupby('GeoID')['ConfirmedDeaths'].agg(['max', 'count']).sort_values(by='max', ascending=False)
    filtered_gdf = gdf[gdf["count"] > min_historical_days]
    countries = list(filtered_gdf.head(nb_geos).index)
    return countries

