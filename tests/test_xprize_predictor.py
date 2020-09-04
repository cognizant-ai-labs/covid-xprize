import os
import unittest

import numpy as np
import pandas as pd

from xprize.xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
EXAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "..", "20200801_20200804_npis.csv")
DATA_URL = os.path.join(ROOT_DIR, "../data", "OxCGRT_latest.csv")
PREDICTOR_27 = os.path.join(FIXTURES_PATH, "20200727_predictor.h5")
PREDICTOR_30 = os.path.join(FIXTURES_PATH, "20200730_predictor.h5")
PREDICTOR_31 = os.path.join(FIXTURES_PATH, "20200731_predictor.h5")
PREDICTIONS_27 = os.path.join(FIXTURES_PATH, "20200727_predictions.csv")
PREDICTIONS_30 = os.path.join(FIXTURES_PATH, "20200730_predictions.csv")
PREDICTIONS_31 = os.path.join(FIXTURES_PATH, "20200731_predictions.csv")

CUTOFF_DATE = np.datetime64("2020-07-31")
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


class TestMultiplicativeEvaluator(unittest.TestCase):

    def test_predict(self):
        predictor = XPrizePredictor(PREDICTOR_31, DATA_URL, CUTOFF_DATE, NPI_COLUMNS)
        start_date = CUTOFF_DATE + np.timedelta64(1, 'D')
        end_date = start_date + np.timedelta64(3, 'D')
        npis_df = pd.read_csv(EXAMPLE_INPUT_FILE,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1")
        pred = predictor.predict(start_date, end_date, npis_df)
        self.assertIsInstance(pred, pd.DataFrame)
        # pred.to_csv(PREDICTIONS_31, index=False)
        # self.assertEqual(pred, 0, "Not the expect prediction")

    def test_train(self):
        predictor = XPrizePredictor(None, DATA_URL, CUTOFF_DATE, NPI_COLUMNS)
        model = predictor.train()
        self.assertIsNotNone(model)
