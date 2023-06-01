# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest
import urllib.request
import gc
from datetime import datetime
from pathlib import Path
import pandas as pd
from covid_xprize import oxford_data

DATA_PATH = Path(__file__).parent / 'test_OxCGRT_data.csv'

class TestPrepareData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = oxford_data.load_oxford_data_trimmed(end_date="2020-12-31")
        data.to_csv(DATA_PATH, index=False)

    def test_load_oxford_data_trimmed(self):
        # Test that loading the trimmed Oxford data loads the correct data timeframe.
        end_date_str = "2020-08-04"
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        df = oxford_data.load_oxford_data_trimmed(end_date_str)
        self.assertEqual(df.Date.max(), end_date)
        # Check that the columns are what we expect them to be.
        self.assertNotIn('GeoID', df.columns) # This function loads the original data with no modifications.
        for NPI_COL in oxford_data.NPI_COLUMNS:
            self.assertIn(NPI_COL, df.columns)
        gc.collect()
    
    def test_create_country_samples(self):
        """Test that the date range of this method conforms to expectations."""
        df = oxford_data.prepare_cases_dataframe(DATA_PATH)
        TRAINING_END_DATE = "2020-11-30"
        DAY_BEFORE_TRAINING_END_DATE = "2020-11-29"
        df = df[df.Date <= TRAINING_END_DATE]
        country_samples = oxford_data.create_country_samples(df, ['France'], 'SmoothNewCasesPer100K', nb_test_days=7, nb_lookback_days=28)
        samples = country_samples['France']
        # Check that the context data window size conforms with the nb_lookback_days.
        # Context should be shape (Samples, Lookback, 1)
        self.assertEqual(samples['X_context'].shape[1], 28)
        self.assertEqual(len(samples['X_context'].squeeze().shape), 2)
        # Y should be shape (Samples, 1)
        self.assertEqual(len(samples['y'].squeeze().shape), 1) # 
        self.assertEqual(samples['y'].shape[0], samples['X_context'].shape[0]) # Samples == Samples

        # Check that the test data goes all the way through the end of the training data.
        gdf = df[df.GeoID == 'France']
        last_day_of_training_cases_data = gdf[gdf.Date == TRAINING_END_DATE           ].SmoothNewCasesPer100K.item()
        penultimate_training_cases_data = gdf[gdf.Date == DAY_BEFORE_TRAINING_END_DATE].SmoothNewCasesPer100K.item()
        self.assertEqual(samples['X_test_context'][-1, -1], penultimate_training_cases_data)
        self.assertEqual(samples['y_test'][-1], last_day_of_training_cases_data)
        gc.collect()

    def test_create_prediction_initial_context_and_action_vectors(self):
        # Test that the context created for the prediction lines up with the last bit of data
        # that comes immediately prior to the prediciton start date. 
        PREDICTIONS_START_DATE = "2020-12-01"
        TRAINING_END_DATE = "2020-11-30"
        df = oxford_data.prepare_cases_dataframe(DATA_PATH)
        context_vectors, action_vectors = oxford_data.create_prediction_initial_context_and_action_vectors(
            df, ['France'], 'SmoothNewCasesPer100K', PREDICTIONS_START_DATE
        )
        prediction_context = context_vectors['France']
        last_day_of_training_cases_data = df[(df.Date == TRAINING_END_DATE) & (df.GeoID == 'France')].SmoothNewCasesPer100K.item()
        self.assertEqual(last_day_of_training_cases_data, prediction_context[-1])
        gc.collect()


if __name__ == '__main__':
    unittest.main()
