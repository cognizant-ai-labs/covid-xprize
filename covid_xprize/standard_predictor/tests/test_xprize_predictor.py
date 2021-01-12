# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest
import urllib.request

import pandas as pd

from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "../../validation/data/2020-09-30_historical_ip.csv")
DATA_FILE = os.path.join(ROOT_DIR, "../data/OxCGRT_latest.csv")
PREDICTOR_WEIGHTS = os.path.join(ROOT_DIR, "../models/trained_model_weights.h5")

START_DATE = "2020-08-01"
END_DATE = "2020-08-04"


class TestXPrizePredictor(unittest.TestCase):

    def test_predict(self):
        predictor = XPrizePredictor(PREDICTOR_WEIGHTS, DATA_FILE)
        pred_df = predictor.predict(START_DATE, END_DATE, EXAMPLE_INPUT_FILE)
        self.assertIsInstance(pred_df, pd.DataFrame)

    def test_train(self):
        predictor = XPrizePredictor(None, DATA_FILE)
        model = predictor.train()
        self.assertIsNotNone(model)
