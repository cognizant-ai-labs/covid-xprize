import os
import unittest

import numpy as np
import pandas as pd

from xprize.xprize_predictor import XPrizePredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
DATA_URL = os.path.join(FIXTURES_PATH, "OxCGRT_latest.csv")

SUBMISSION_DATE = np.datetime64("2020-07-31")
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

    @classmethod
    def setUpClass(cls):
        # Executed once before tests in this class are run
        latest_df = pd.read_csv(DATA_URL,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                error_bad_lines=False)
        # Handle regions
        latest_df["RegionName"].fillna('', inplace=True)
        # Replace CountryName by CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["CountryName"] = np.where(latest_df["RegionName"] == '',
                                            latest_df["CountryName"],
                                            latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
        # Take a snapshot on submission date
        snapshot_df = latest_df[latest_df.Date <= SUBMISSION_DATE]
        cls._latest_df = latest_df
        cls._snapshot_df = snapshot_df

    def setUp(self):
        # Executed before each test
        pass

    def test_simple_roll_out(self):
        cls = self.__class__
        predictor = XPrizePredictor()
        # Forecast for the first day after submission
        day_1 = SUBMISSION_DATE + np.timedelta64(1, 'D')
        day_1_npis_df = cls._snapshot_df[cls._snapshot_df.Date <= day_1]
        pred = predictor.submission_predict(day_1, day_1_npis_df)
        self.assertEqual(pred, 0, "Not the expect prediction")
