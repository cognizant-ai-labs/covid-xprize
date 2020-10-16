import os
import unittest
import urllib.request

import pandas as pd

from validation.scenario_generator import generate_scenario

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
DATA_FILE = os.path.join(FIXTURES_PATH, "OxCGRT_latest.csv")
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"


def _get_dataset():
    # Download and cache the raw data file if it doesn't exist
    if not os.path.exists(DATA_FILE):
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    latest_df = pd.read_csv(DATA_FILE,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    return latest_df


class TestScenarioGenerator(unittest.TestCase):

    def test_generate_scenario(self):
        latest_df = _get_dataset()
        start_date_str = "2020-08-01"
        end_date_str = "2020-08-4"
        countries = ["Italy"]
        scenario_df = generate_scenario(start_date_str, end_date_str, latest_df, countries)
        self.assertIsNotNone(scenario_df)
        self.assertEqual(1, len(scenario_df.CountryName.unique()), "Expected only 1 country")
        self.assertEqual("Italy", scenario_df.CountryName.unique()[0], "Not the requested country")
        # Inception is 1/1/2020
        self.assertEqual(217, len(scenario_df), "Expected the number of days between inception and end date")
