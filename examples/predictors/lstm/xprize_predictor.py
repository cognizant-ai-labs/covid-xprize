# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os

# Suppress noisy Tensorflow debug logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# noinspection PyPep8Naming
import keras.backend as K
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.constraints import Constraint
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Lambda
from keras.models import Model
from keras.models import load_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")

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
               'H3_Contact tracing']

CONTEXT_COLUMNS = ['CountryName',
                   'RegionName',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population']
NB_LOOKBACK_DAYS = 21
NB_TEST_DAYS = 14
WINDOW_SIZE = 7
US_PREFIX = "United States / "
NUM_TRIALS = 1
LSTM_SIZE = 32
MAX_NB_COUNTRIES = 20


class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)


# Functions to be used for lambda layers in model
def _combine_r_and_d(x):
    r, d = x
    return r * (1. - d)


class XPrizePredictor(object):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self, path_to_model_weights, data_url, cutoff_date_str):
        if path_to_model_weights:
            nb_context = 1 # Only time series of new cases rate is used as context
            nb_action = len(NPI_COLUMNS)
            self.predictor, _ = self._construct_model(nb_context=nb_context,
                                                      nb_action=nb_action,
                                                      lstm_size=LSTM_SIZE,
                                                      nb_lookback_days=NB_LOOKBACK_DAYS)
            self.predictor.load_weights(path_to_model_weights)
        cutoff_date = pd.to_datetime(cutoff_date_str, format='%Y-%m-%d')
        self.df = self._prepare_dataframe(data_url, cutoff_date)
        self.countries = self.df.CountryName.unique()
        self.country_samples = self._create_country_samples(self.df, self.countries)

    def predict(self,
                start_date_str: str,
                end_date_str: str,
                npis_csv: str) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        # end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        # Load the npis into a DataFrame, handling regions
        npis_df = self._load_original_data(npis_csv)

        # Prepare the output
        forecast = {"CountryName": [],
                    "RegionName": [],
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
            npis_sequence = np.array(cnpis_df[NPI_COLUMNS])

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
                # Split CountryName back into CountryName and RegionName
                c_split = c.split(" / ")
                country = c_split[0]
                region = c_split[1] if len(c_split) > 1 else np.NaN
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                # forecast["ConfirmedCases"].append(pred_total_cases[i])
                forecast["PredictedDailyNewCases"].append(pred)

        forecast_df = pd.DataFrame.from_dict(forecast)
        return forecast_df

    def _prepare_dataframe(self, data_url: str, cutoff_date: pd.Timestamp) -> (pd.DataFrame, pd.DataFrame):
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :param data_url: the url containing the original data
        :param cutoff_date: last date to use when loading the data
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df1 = self._load_original_data(data_url, cutoff_date)

        # Additional context df (e.g Population for each country)
        df2 = self._load_additional_context_df()

        # Merge the 2 DataFrames
        df = df1.merge(df2, on=['CountryName'], how='left', suffixes=('', '_y'))

        # Drop countries with no population data
        df.dropna(subset=['Population'], inplace=True)

        #  Keep only needed columns
        columns = CONTEXT_COLUMNS + NPI_COLUMNS
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

    def _load_original_data(self, data_url, cutoff_date=None):
        latest_df = pd.read_csv(data_url,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                error_bad_lines=False,
                                low_memory=False)
        # Handle regions.
        # Replace CountryName by CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["CountryName"] = np.where(latest_df["RegionName"].isnull(),
                                            latest_df["CountryName"],
                                            latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
        # Take a snapshot on cutoff_date
        if cutoff_date:
            return latest_df[latest_df.Date <= cutoff_date]
        else:
            return latest_df

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
        for npi_column in NPI_COLUMNS:
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
        context_column = 'PredictionRatio'
        action_columns = NPI_COLUMNS
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
            for d in range(NB_LOOKBACK_DAYS, nb_total_days):
                context_samples.append(context_data[d - NB_LOOKBACK_DAYS:d])
                action_samples.append(action_data[d - NB_LOOKBACK_DAYS:d])
                outcome_samples.append(outcome_data[d])
            if len(outcome_samples) > 0:
                X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
                X_action = np.stack(action_samples, axis=0)
                y = np.stack(outcome_samples, axis=0)
                country_samples[c] = {
                    'X_context': X_context,
                    'X_action': X_action,
                    'y': y,
                    'X_train_context': X_context[:-NB_TEST_DAYS],
                    'X_train_action': X_action[:-NB_TEST_DAYS],
                    'y_train': y[:-NB_TEST_DAYS],
                    'X_test_context': X_context[-NB_TEST_DAYS:],
                    'X_test_action': X_action[-NB_TEST_DAYS:],
                    'y_test': y[-NB_TEST_DAYS:],
                }
        return country_samples

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

    def train(self):
        print("Creating numpy arrays for Keras for each country...")
        countries = self._most_affected_countries(self.df, MAX_NB_COUNTRIES, NB_LOOKBACK_DAYS)
        country_samples = self._create_country_samples(self.df, countries)
        print("Numpy arrays created")

        # Aggregate data for training
        all_X_context_list = [country_samples[c]['X_train_context']
                              for c in country_samples]
        all_X_action_list = [country_samples[c]['X_train_action']
                             for c in country_samples]
        all_y_list = [country_samples[c]['y_train']
                      for c in country_samples]
        X_context = np.concatenate(all_X_context_list)
        X_action = np.concatenate(all_X_action_list)
        y = np.concatenate(all_y_list)

        # Clip outliers
        MIN_VALUE = 0.
        MAX_VALUE = 2.
        X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
        y = np.clip(y, MIN_VALUE, MAX_VALUE)

        # Aggregate data for testing only on top countries
        test_all_X_context_list = [country_samples[c]['X_train_context']
                                   for c in countries]
        test_all_X_action_list = [country_samples[c]['X_train_action']
                                  for c in countries]
        test_all_y_list = [country_samples[c]['y_train']
                           for c in countries]
        test_X_context = np.concatenate(test_all_X_context_list)
        test_X_action = np.concatenate(test_all_X_action_list)
        test_y = np.concatenate(test_all_y_list)

        test_X_context = np.clip(test_X_context, MIN_VALUE, MAX_VALUE)
        test_y = np.clip(test_y, MIN_VALUE, MAX_VALUE)

        # Run full training several times to find best model
        # and gather data for setting acceptance threshold
        models = []
        train_losses = []
        val_losses = []
        test_losses = []
        for t in range(NUM_TRIALS):
            print('Trial', t)
            X_context, X_action, y = self._permute_data(X_context, X_action, y, seed=t)
            model, training_model = self._construct_model(nb_context=X_context.shape[-1],
                                                          nb_action=X_action.shape[-1],
                                                          lstm_size=LSTM_SIZE,
                                                          nb_lookback_days=NB_LOOKBACK_DAYS)
            history = self._train_model(training_model, X_context, X_action, y, epochs=1000, verbose=0)
            top_epoch = np.argmin(history.history['val_loss'])
            train_loss = history.history['loss'][top_epoch]
            val_loss = history.history['val_loss'][top_epoch]
            test_loss = training_model.evaluate([test_X_context, test_X_action], [test_y])
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            models.append(model)
            print('Train Loss:', train_loss)
            print('Val Loss:', val_loss)
            print('Test Loss:', test_loss)

        # Gather test info
        country_indeps = []
        country_predss = []
        country_casess = []
        for model in models:
            country_indep, country_preds, country_cases = self._lstm_get_test_rollouts(model,
                                                                                       self.df,
                                                                                       countries,
                                                                                       country_samples)
            country_indeps.append(country_indep)
            country_predss.append(country_preds)
            country_casess.append(country_cases)

        # Compute cases mae
        test_case_maes = []
        for m in range(len(models)):
            total_loss = 0
            for c in countries:
                true_cases = np.sum(np.array(self.df[self.df.CountryName == c].NewCases)[-NB_TEST_DAYS:])
                pred_cases = np.sum(country_casess[m][c][-NB_TEST_DAYS:])
                total_loss += np.abs(true_cases - pred_cases)
            test_case_maes.append(total_loss)

        # Select best model
        best_model = models[np.argmin(test_case_maes)]
        self.predictor = best_model
        print("Done")
        return best_model

    def _most_affected_countries(self, df, nb_countries, min_historical_days):
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

    # Shuffling data prior to train/val split
    def _permute_data(self, X_context, X_action, y, seed=301):
        np.random.seed(seed)
        p = np.random.permutation(y.shape[0])
        X_context = X_context[p]
        X_action = X_action[p]
        y = y[p]
        return X_context, X_action, y

    # Construct model
    def _construct_model(self, nb_context, nb_action, lstm_size=32, nb_lookback_days=21):

        # Create context encoder
        context_input = Input(shape=(nb_lookback_days, nb_context),
                              name='context_input')
        x = LSTM(lstm_size, name='context_lstm')(context_input)
        context_output = Dense(units=1,
                               activation='softplus',
                               name='context_dense')(x)

        # Create action encoder
        # Every aspect is monotonic and nonnegative except final bias
        action_input = Input(shape=(nb_lookback_days, nb_action),
                             name='action_input')
        x = LSTM(units=lstm_size,
                 kernel_constraint=Positive(),
                 recurrent_constraint=Positive(),
                 bias_constraint=Positive(),
                 return_sequences=False,
                 name='action_lstm')(action_input)
        action_output = Dense(units=1,
                              activation='sigmoid',
                              kernel_constraint=Positive(),
                              name='action_dense')(x)

        # Create prediction model
        model_output = Lambda(_combine_r_and_d, name='prediction')(
            [context_output, action_output])
        model = Model(inputs=[context_input, action_input],
                      outputs=[model_output])
        model.compile(loss='mae', optimizer='adam')

        # Create training model, which includes loss to measure
        # variance of action_output predictions
        training_model = Model(inputs=[context_input, action_input],
                               outputs=[model_output])
        training_model.compile(loss='mae',
                               optimizer='adam')

        return model, training_model

    # Train model
    def _train_model(self, training_model, X_context, X_action, y, epochs=1, verbose=0):
        early_stopping = EarlyStopping(patience=20,
                                       restore_best_weights=True)
        history = training_model.fit([X_context, X_action], [y],
                                     epochs=epochs,
                                     batch_size=32,
                                     validation_split=0.1,
                                     callbacks=[early_stopping],
                                     verbose=verbose)
        return history

    # Functions for computing test metrics
    def _lstm_roll_out_predictions(self, model, initial_context_input, initial_action_input, future_action_sequence):
        nb_test_days = future_action_sequence.shape[0]
        pred_output = np.zeros(nb_test_days)
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        for d in range(nb_test_days):
            action_input[:, :-1] = action_input[:, 1:]
            action_input[:, -1] = future_action_sequence[d]
            pred = model.predict([context_input, action_input])
            pred_output[d] = pred
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1] = pred
        return pred_output

    def _lstm_get_test_rollouts(self, model, df, top_countries, country_samples):
        country_indep = {}
        country_preds = {}
        country_cases = {}
        for c in top_countries:
            X_test_context = country_samples[c]['X_test_context']
            X_test_action = country_samples[c]['X_test_action']
            country_indep[c] = model.predict([X_test_context, X_test_action])

            initial_context_input = country_samples[c]['X_test_context'][0]
            initial_action_input = country_samples[c]['X_test_action'][0]
            y_test = country_samples[c]['y_test']

            nb_test_days = y_test.shape[0]
            nb_actions = initial_action_input.shape[-1]

            future_action_sequence = np.zeros((nb_test_days, nb_actions))
            future_action_sequence[:nb_test_days] = country_samples[c]['X_test_action'][:, -1, :]
            current_action = country_samples[c]['X_test_action'][:, -1, :][-1]
            future_action_sequence[14:] = current_action
            preds = self._lstm_roll_out_predictions(model,
                                                    initial_context_input,
                                                    initial_action_input,
                                                    future_action_sequence)
            country_preds[c] = preds

            prev_confirmed_cases = np.array(
                df[df.CountryName == c].ConfirmedCases)[:-nb_test_days]
            prev_new_cases = np.array(
                df[df.CountryName == c].NewCases)[:-nb_test_days]
            initial_total_cases = prev_confirmed_cases[-1]
            pop_size = np.array(df[df.CountryName == c].Population)[0]

            _, pred_new_cases = self._convert_ratios_to_total_cases(
                preds, WINDOW_SIZE, prev_new_cases, initial_total_cases, pop_size)
            country_cases[c] = pred_new_cases

        return country_indep, country_preds, country_cases
