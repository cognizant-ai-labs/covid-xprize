# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest
import urllib.request

import pandas as pd

from examples.predictors.lstm.xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
EXAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "../../../validation/data/2020-08-01_2020-08-04_ip.csv")
DATA_FILE = os.path.join(FIXTURES_PATH, "OxCGRT_latest.csv")
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
PREDICTOR_27 = os.path.join(FIXTURES_PATH, "pred27", "predictor.h5")
PREDICTOR_30 = os.path.join(FIXTURES_PATH, "pred30", "predictor.h5")
PREDICTOR_31 = os.path.join(FIXTURES_PATH, "pred31", "predictor.h5")
PREDICTIONS_27 = os.path.join(FIXTURES_PATH, "pred27", "20200801_20200804_predictions.csv")
PREDICTIONS_30 = os.path.join(FIXTURES_PATH, "pred30", "20200801_20200804_predictions.csv")
PREDICTIONS_31 = os.path.join(FIXTURES_PATH, "pred31", "20200801_20200804_predictions.csv")

CUTOFF_DATE = "2020-07-31"
START_DATE = "2020-08-01"
END_DATE = "2020-08-04"


class TestMultiplicativeEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Download and cache the raw data file if it doesn't exist
        if not os.path.exists(DATA_FILE):
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    def test_predict(self):
        predictor = XPrizePredictor(PREDICTOR_31, DATA_FILE, CUTOFF_DATE)
        pred_df = predictor.predict(START_DATE, END_DATE, EXAMPLE_INPUT_FILE)
        self.assertIsInstance(pred_df, pd.DataFrame)

    def test_train(self):
        predictor = XPrizePredictor(None, DATA_FILE, CUTOFF_DATE)
        model = predictor.train()
        self.assertIsNotNone(model)
