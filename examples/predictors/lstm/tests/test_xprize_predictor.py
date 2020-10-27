# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest
import urllib.request

import pandas as pd

from examples.predictors.lstm.xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
EXAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "../../../../validation/data/2020-09-30_historical_ip.csv")
DATA_FILE = os.path.join(FIXTURES_PATH, "OxCGRT_latest.csv")
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
PREDICTOR_WEIGHTS = os.path.join(FIXTURES_PATH, "trained_model_weights_for_tests.h5")

CUTOFF_DATE = "2020-07-31"
START_DATE = "2020-08-01"
END_DATE = "2020-08-04"


class TestXPrizePredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Download and cache the raw data
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    def test_predict(self):
        predictor = XPrizePredictor(PREDICTOR_WEIGHTS, DATA_FILE, CUTOFF_DATE)
        pred_df = predictor.predict(START_DATE, END_DATE, EXAMPLE_INPUT_FILE)
        self.assertIsInstance(pred_df, pd.DataFrame)

    def test_train(self):
        predictor = XPrizePredictor(None, DATA_FILE, CUTOFF_DATE)
        model = predictor.train()
        self.assertIsNotNone(model)
