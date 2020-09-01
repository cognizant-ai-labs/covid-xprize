import os

import pandas as pd
import numpy as np

import keras.backend as K
from keras.constraints import Constraint
from keras.models import load_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")

CONTEXT_COLUMNS = ['CountryName',
                   'CountryCode',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population']
WINDOW_SIZE = 7
US_PREFIX = "United States / "


class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)


class XPrizePredictor(object):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self, path_to_model, snapshot_df, npi_columns):
        self.predictor = load_model(path_to_model, custom_objects={"Positive": Positive})
        self.npi_columns = npi_columns
        self.df = self._prepare_dataframe(snapshot_df)
        self.countries = self.df.CountryName.unique()
        self.country_samples = self._create_country_samples(self.df, self.countries)

    def submission_predict(self,
                           start_date: np.datetime64,
                           end_date: np.datetime64,
                           npis: pd.DataFrame) -> pd.DataFrame:
        """
        Makes a prediction of daily confirmed cases for a given day.
        :param start_date: the day from which to start making predictions
        :param end_date: the day on which to stop making predictions
        :param npis: the actual npis between start_date and end_date
        :return: a Pandas DataFrame containing the prediction [`CountryName`, `Date`, `NewCases]
        """
        return self.simple_roll_out(start_date, end_date, npis)

    def _prepare_dataframe(self,
                           measures_by_country_df=None) -> (pd.DataFrame, pd.DataFrame):
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :param measures_by_country_df: the DataFrame containing the initial data
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df1 = measures_by_country_df

        # Additional context df (e.g Population for each country)
        df2 = self._load_additional_context_df()

        # Merge the 2 DataFrames
        df = df1.merge(df2, on=['CountryName'], how='left', suffixes=('', '_y'))

        # Drop countries with no population data
        df.dropna(subset=['Population'], inplace=True)

        #  Keep only needed columns
        columns = CONTEXT_COLUMNS + self.npi_columns
        df = df[columns]

        # Fill in missing values
        self._fill_missing_values(df)

        # Compute number of new cases and deaths each day
        df['NewCases'] = df.groupby('CountryName').ConfirmedCases.diff().fillna(0)
        df['NewDeaths'] = df.groupby('CountryName').ConfirmedDeaths.diff().fillna(0)

        # Replace negative values (which do not make sense for these columns) with 0
        df['NewCases'] = df['NewCases'].clip(lower=0)
        df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

        # Compute smoothed versions of new cases and deaths each day
        df['SmoothNewCases'] = df.groupby('CountryName')['NewCases'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
        df['SmoothNewDeaths'] = df.groupby('CountryName')['NewDeaths'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

        # Compute percent change in new cases and deaths each day
        df['CaseRatio'] = df.groupby('CountryName').SmoothNewCases.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1
        df['DeathRatio'] = df.groupby('CountryName').SmoothNewDeaths.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1

        # # Remove all rows with too few cases
        # df.drop(df[df.ConfirmedCases < MIN_CASES].index, inplace=True)

        # Add column for proportion of population infected
        df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

        # Create column of value to predict
        df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

        return df

    def _fill_missing_values(self, df):
        """
        # Fill missing values by interpolation, ffill, and filling NaNs
        :param df: Dataframe to be filled
        """
        df.update(df.groupby('CountryName').ConfirmedCases.apply(
            lambda group: group.interpolate(limit_area='inside')))
        df.dropna(subset=['ConfirmedCases'], inplace=True)
        df.update(df.groupby('CountryName').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)
        for npi_column in self.npi_columns:
            df.update(df.groupby('CountryName')[npi_column].ffill().fillna(0))

    @staticmethod
    def _load_additional_context_df():
        # File containing the population for each country
        additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                            usecols=['CountryName', 'Population'])

        # US states population
        additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                              usecols=['NAME', 'POPESTIMATE2019'])
        # Rename the columns to match measures_df ones
        additional_us_states_df.rename(columns={'NAME': 'CountryName',
                                                'POPESTIMATE2019': 'Population'},
                                       inplace=True)
        # Prefix with country name to match measures_df
        additional_us_states_df['CountryName'] = US_PREFIX + additional_us_states_df['CountryName']

        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_us_states_df)

        # UK population
        additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_uk_df)

        return additional_context_df

    def _create_country_samples(self, df: pd.DataFrame, countries: list) -> dict:
        """
        For each country, creates numpy arrays for Keras
        :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
        :param countries: a list of country names
        :return: a dictionary of train and test sets, for each specified country
        """
        nb_lookback_days = 21
        nb_test_days = 14
        context_column = 'PredictionRatio'
        action_columns = self.npi_columns
        outcome_column = 'PredictionRatio'
        country_samples = {}
        for c in countries:
            cdf = df[df.CountryName == c]
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

    def simple_roll_out(self,
                        start_date,
                        end_date,
                        npis_df):
        # Prepare the output
        forecast = {"CountryName": [],
                    "Date": [],
                    "PredictedDailyNewCases": []}

        # For each country, each region
        for c in self.countries:
            cdf = self.df[self.df.CountryName == c]
            cdf = cdf[cdf.ConfirmedCases.notnull()]
            initial_context_input = self.country_samples[c]['X_test_context'][-1]
            initial_action_input = self.country_samples[c]['X_test_action'][-1]

            # Predictions with passed npis
            cnpis_df = npis_df[npis_df.CountryName == c]
            npis_sequence = np.array(cnpis_df[self.npi_columns])

            # Get the predictions with the passed NPIs
            preds = self._roll_out_predictions(self.predictor,
                                               initial_context_input,
                                               initial_action_input,
                                               npis_sequence)

            # Gather info to convert to total cases
            prev_confirmed_cases = np.array(cdf.ConfirmedCases)
            prev_new_cases = np.array(cdf.NewCases)
            initial_total_cases = prev_confirmed_cases[-1]
            pop_size = np.array(cdf.Population)[-1]  # Population size doesn't change over time

            # Compute predictor's forecast
            pred_total_cases, pred_new_cases = self._convert_ratios_to_total_cases(
                preds,
                WINDOW_SIZE,
                prev_new_cases,
                initial_total_cases,
                pop_size)
            # OPTIONAL: Smooth out pred_new_cases
            # # If window size is 7, take the previous 6 new cases so we start doing a 7 day moving average for
            # # the first pred new cases
            # temp_pred_new_cases = list(prev_new_cases[-(WINDOW_SIZE-1):]) + pred_new_cases
            # smooth_pred_new_cases = self._smooth_case_list(temp_pred_new_cases, WINDOW_SIZE)
            # # Get rid of the first window_size - 1 NaN values where
            # # there was not enough data to compute a moving average
            # pred_new_cases = smooth_pred_new_cases[WINDOW_SIZE-1:]

            # Append forecast data to results to return
            for i, pred in enumerate(pred_new_cases):
                forecast["CountryName"].append(c)
                current_date = start_date + np.timedelta64(i, 'D')
                forecast["Date"].append(current_date)
                # forecast["ConfirmedCases"].append(pred_total_cases[i])
                forecast["PredictedDailyNewCases"].append(pred)

        forecast_df = pd.DataFrame.from_dict(forecast)
        return forecast_df

    # Function for performing roll outs into the future
    @staticmethod
    def _roll_out_predictions(predictor, initial_context_input, initial_action_input, future_action_sequence):
        nb_roll_out_days = future_action_sequence.shape[0]
        pred_output = np.zeros(nb_roll_out_days)
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        for d in range(nb_roll_out_days):
            action_input[:, :-1] = action_input[:, 1:]
            # Use the passed actions
            action_sequence = future_action_sequence[d]
            action_input[:, -1] = action_sequence
            pred = predictor.predict([context_input, action_input])
            pred_output[d] = pred
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1] = pred
        return pred_output

    # Functions for converting predictions back to number of cases
    @staticmethod
    def _convert_ratio_to_new_cases(ratio,
                                    window_size,
                                    prev_new_cases_list,
                                    prev_pct_infected):
        return (ratio * (1 - prev_pct_infected) - 1) * \
               (window_size * np.mean(prev_new_cases_list[-window_size:])) \
               + prev_new_cases_list[-window_size]

    def _convert_ratios_to_total_cases(self,
                                       ratios,
                                       window_size,
                                       prev_new_cases,
                                       initial_total_cases,
                                       pop_size):
        new_total_cases = []
        new_new_cases = []
        prev_new_cases_list = list(prev_new_cases)
        curr_total_cases = initial_total_cases
        for ratio in ratios:
            new_cases = self._convert_ratio_to_new_cases(ratio,
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
            new_total_cases.append(curr_total_cases)
        return new_total_cases, new_new_cases

    @staticmethod
    def _smooth_case_list(case_list, window):
        return pd.Series(case_list).rolling(window).mean().to_numpy()
