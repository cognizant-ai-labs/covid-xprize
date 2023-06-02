# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
from pathlib import Path
import unittest
import urllib.request

import pandas as pd

from covid_xprize.examples.predictors.lstm.xprize_predictor import XPrizePredictor
from covid_xprize.oxford_data import load_oxford_data_trimmed

ROOT_DIR = Path(__file__).parent
FIXTURES_PATH = ROOT_DIR / 'fixtures'
EXAMPLE_INPUT_FILE = (ROOT_DIR / "../../../../validation/data/2020-09-30_historical_ip.csv").absolute()
DATA_FILE = FIXTURES_PATH / "OxCGRT_trimmed.csv"
PREDICTOR_WEIGHTS = FIXTURES_PATH / "trained_model_weights_for_tests.h5"

TRAINING_START_DATE = "2020-06-01"
TRAINING_END_DATE = "2020-07-31"
START_DATE = "2020-08-01"
END_DATE = "2020-08-04"


class TestXPrizePredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        FIXTURES_PATH.mkdir(exist_ok=True)
        if not DATA_FILE.exists():
            df = load_oxford_data_trimmed(end_date=TRAINING_END_DATE, start_date=TRAINING_START_DATE)
            df.to_csv(DATA_FILE, index=False)

    def test_train_and_predict(self):
        predictor = XPrizePredictor(None, DATA_FILE)
        model = predictor.train(num_epochs=2)
        model.save_weights(PREDICTOR_WEIGHTS)
        self.assertIsNotNone(model)

        predictor = XPrizePredictor(PREDICTOR_WEIGHTS, DATA_FILE)
        pred_df = predictor.predict(START_DATE, END_DATE, EXAMPLE_INPUT_FILE)
        self.assertIsInstance(pred_df, pd.DataFrame)
