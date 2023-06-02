# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import urllib.request
from typing import Union

import numpy as np
import pandas as pd
from keras.models import Model

from covid_xprize.examples.predictors.conditional_lstm.conditional_lstm_model import construct_conditional_lstm_model
from covid_xprize.examples.predictors.conditional_lstm.train_predictor import train_predictor, construct_model
from covid_xprize.oxford_data.npi_static import NPI_COLUMNS
from covid_xprize.oxford_data import prepare_cases_dataframe, load_ips_file, \
                                     create_prediction_initial_context_and_action_vectors, \
                                     convert_smooth_cases_per_100K_to_new_cases


WINDOW_SIZE = 7
NB_LOOKBACK_DAYS = 21
NUM_TRIALS = 20
LSTM_SIZE = 16
NB_TEST_DAYS = 28
NB_TRAINING_DAYS = 365
NB_TRAINING_GEOS = None
NB_TESTING_GEOS = None
NUM_EPOCHS = 1000
LSTM_SIZE = 16
CONTEXT_COLUMN = 'SmoothNewCasesPer100K'


class ConditionalXPrizePredictor(object):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self, path_to_model_weights, data_url):
        if path_to_model_weights:
            nb_context = 1  # Only time series of new cases rate is used as context
            nb_action = len(NPI_COLUMNS)
            self.predictor = construct_model(nb_context=nb_context,
                                            nb_action=nb_action,
                                            lstm_size=LSTM_SIZE,
                                            nb_lookback_days=NB_LOOKBACK_DAYS)
            self.predictor.load_weights(path_to_model_weights)

        self.df = prepare_cases_dataframe(data_url, threshold_min_cases=True)

    def train(self,
              return_results=False,
              nb_training_geos: int = NB_TRAINING_DAYS,
              nb_testing_geos: int = NB_TESTING_GEOS,
              nb_trials: int = NUM_TRIALS,
              nb_epochs: int = NUM_EPOCHS,) -> Union[Model, tuple[Model, dict]]:
        best_model, results_df = train_predictor(
            training_data=self.df,
            nb_lookback_days=NB_LOOKBACK_DAYS,
            nb_training_days=NB_TRAINING_DAYS,
            nb_test_days=NB_TEST_DAYS,
            nb_training_geos=nb_training_geos,
            nb_testing_geos=nb_testing_geos,
            nb_trials=nb_trials,
            nb_epochs=nb_epochs,
            lstm_size=LSTM_SIZE,
        )
        if return_results:
            return best_model, results_df
        return best_model

    def predict(self,
                start_date_str: str,
                end_date_str: str,
                path_to_ips_file: str) -> pd.DataFrame:
        # Access the Oxford dataset for the cases & other context data for this time.
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        nb_days = (end_date - start_date).days + 1 # Inclusive interval

        # Load the npis into a DataFrame, handling regions
        npis_df = load_ips_file(path_to_ips_file)

        # Prepare the output
        forecast = {"CountryName": [],
                    "RegionName": [],
                    "Date": [],
                    "PredictedDailyNewCases": []}

        # For each requested geo
        geos = npis_df.GeoID.unique()

        # Prepare context vectors.
        initial_context, initial_action = create_prediction_initial_context_and_action_vectors(
            self.df,
            geos,
            CONTEXT_COLUMN,
            start_date,
            NB_LOOKBACK_DAYS,
        )
        for g in geos:
            cdf = self.df[self.df.GeoID == g]
            if len(cdf) == 0:
                # we don't have historical data for this geo: return zeroes
                pred_new_cases = [0] * nb_days
                geo_start_date = start_date
            else:
                last_known_date = cdf.Date.max()
                # Start predicting from start_date, unless there's a gap since last known date
                ONE_DAY = np.timedelta64(1, 'D')
                geo_start_date = min(last_known_date + ONE_DAY, start_date)
                npis_gdf = npis_df[(npis_df.Date >= geo_start_date) & (npis_df.Date <= end_date)]
                pred_new_cases = self._get_new_cases_preds(cdf, g, npis_gdf, initial_context[g], initial_action[g])

            # Append forecast data to results to return
            country = npis_df[npis_df.GeoID == g].iloc[0].CountryName
            region = npis_df[npis_df.GeoID == g].iloc[0].RegionName
            for i, pred in enumerate(pred_new_cases):
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = geo_start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                forecast["PredictedDailyNewCases"].append(pred)

        forecast_df = pd.DataFrame.from_dict(forecast)
        # Return only the requested predictions
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]

    def _get_new_cases_preds(self, c_df: pd.DataFrame, g: str, npis_df, initial_context_input, initial_action_input):
        cdf = c_df[c_df.ConfirmedCases.notnull()]
        # Predictions with passed npis
        cnpis_df = npis_df[npis_df.GeoID == g]
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
        pred_new_cases = convert_smooth_cases_per_100K_to_new_cases(
            preds,
            WINDOW_SIZE,
            prev_new_cases,
            pop_size
        )

        return pred_new_cases

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

