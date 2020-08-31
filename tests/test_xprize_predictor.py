import unittest

import numpy as np
import pandas as pd

DATA_URL = "tests/fixtures/OxCGRT_latest.csv"
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
        pass
