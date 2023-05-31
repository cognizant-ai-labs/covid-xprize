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


# A link to the Oxford data set.
OXFORD_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker-legacy/main/legacy_data_202207/OxCGRT_latest.csv'


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
                        on_bad_lines='skip')


def load_original_oxford_data() -> pd.DataFrame:
    """Loads the original Oxford dataset with no preprocessing."""
    return _load_geospatial_df(OXFORD_DATA_URL)


def load_oxford_data_trimmed(end_date: str) -> pd.DataFrame:
    """
    Loads the very original Oxford dataset and removes dates after the given end_date.

    :param end_date: The final date of data to return, format YYYY-MM-DDD.
    """
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    df = load_original_oxford_data()
    df = df[(df.Date <= end_date)]
    return df


def _add_geoid_column(df: pd.DataFrame) -> None:
    """Add GeoID column that combines CountryName and RegionName for easier manipulation of data."""
    # GeoID is CountryName / RegionName
    # np.where usage: if A then B else C
    df["GeoID"] = np.where( df["RegionName"].isnull(),
                            df["CountryName"],
                            df["CountryName"] + ' / ' + df["RegionName"])


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


def prepare_cases_dataframe(data_url: str, threshold_min_cases=False) -> pd.DataFrame:
    """
    Loads the cases dataset from the given file, cleans it, and computes cases columns.
    :param data_url: the url containing the original data
    :return: a Pandas DataFrame with the historical data
    """
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

    # Remove all rows with too few cases
    if threshold_min_cases:
        df.drop(df[df.ConfirmedCases < MIN_CASES].index, inplace=True)

    # Add column for proportion of population infected
    df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

    # Create column of value to predict
    df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

    # Create column of for cases per 100K
    df['SmoothNewCasesPer100K'] = df['SmoothNewCases'] / (df['Population'] / 100_000)

    return df


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
        context_vectors[c] = country_samples[c]['X_test_context'][-1]
        action_vectors[c] =  country_samples[c]['X_test_action'][-1]
    return context_vectors, action_vectors


def most_affected_countries(df, nb_countries, min_historical_days):
    """
    Returns the list of most affected countries, in terms of confirmed deaths.
    :param df: the data frame containing the historical data
    :param nb_countries: the number of countries to return
    :param min_historical_days: the minimum days of historical data the countries must have
    :return: a list of country names of size nb_countries if there were enough, and otherwise a list of all the
    country names that have at least min_look_back_days data points.
    """
    # By default use most affected countries with enough history
    gdf = df.groupby('CountryName')['ConfirmedDeaths'].agg(['max', 'count']).sort_values(by='max', ascending=False)
    filtered_gdf = gdf[gdf["count"] > min_historical_days]
    countries = list(filtered_gdf.head(nb_countries).index)
    return countries


def convert_smooth_cases_per_100K_to_new_cases(smooth_cases_per_100K,
                                               window_size,
                                               prev_new_cases_list,
                                               pop_size):
    return (((window_size * pop_size) / 100000.) * smooth_cases_per_100K \
            - np.sum(prev_new_cases_list[-(window_size-1):])).clip(min=0.0)


def convert_ratio_to_new_cases(ratio,
                                window_size,
                                prev_new_cases_list,
                                prev_pct_infected):
    return (ratio * (1 - prev_pct_infected) - 1) * \
            (window_size * np.mean(prev_new_cases_list[-window_size:])) \
            + prev_new_cases_list[-window_size]

