# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest
import urllib.request

import numpy as np
import pandas as pd

from validation.scenario_generator import generate_scenario, NPI_COLUMNS, MIN_NPIS, MAX_NPIS

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
DATA_FILE = os.path.join(FIXTURES_PATH, "OxCGRT_latest.csv")
DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

# Sets each NPI to level 1
ONE_NPIS = [1] * len(NPI_COLUMNS)


def _get_dataset():
    # Download and cache the raw data file if it doesn't exist
    if not os.path.exists(DATA_FILE):
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    latest_df = pd.read_csv(DATA_FILE,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    # Fill any missing NPIs by assuming they are the same as previous day, or 0 if none is available
    latest_df.update(latest_df.groupby(['CountryName', 'RegionName'])[NPI_COLUMNS].ffill().fillna(0))
    return latest_df


class TestScenarioGenerator(unittest.TestCase):
    """
    Tests generating different NPI scenarios.

    Definitions:
    I = inception date = 2020-01-01 (earliest available data)
    LK = last known date (for each country) in the latest available data
    S = start date of the scenario
    E = end date of the scenario

    Time wise, the following kind of scenarios can be applied:
    1. Counterfactuals: I____S_____E____LK where E can equal LK
    2. Future:          I____S_____LK___E where S can equal LK
    3. Mind the gap:    I____LK    S____E

    For each case, we check each type of scenario: freeze, MIN, MAX, custom

    Scenarios can be applied to: 1 country, several countries, all countries
    """

    @classmethod
    def setUpClass(cls):
        # Load the csv data only once
        cls.latest_df = _get_dataset()

    def test_generate_scenario_counterfactual_freeze(self):
        # Simulate Italy did not enter full lockdown on Mar 20, but instead waited 1 week before changing its NPIs
        before_day = pd.to_datetime("2020-03-19", format='%Y-%m-%d')
        frozen_npis_df = self.latest_df[(self.latest_df.CountryName == "Italy") &
                                        (self.latest_df.Date == before_day)][NPI_COLUMNS].reset_index(drop=True)
        frozen_npis = list(frozen_npis_df.values[0])
        self._check_counterfactual("Freeze", frozen_npis)

    def test_generate_scenario_counterfactual_min(self):
        # Simulate Italy lifted all NPIs for a period
        self._check_counterfactual("MIN", MIN_NPIS)

    def test_generate_scenario_counterfactual_max(self):
        # Simulate Italy maxed out all NPIs for a period
        self._check_counterfactual("MAX", MAX_NPIS)

    def test_generate_scenario_counterfactual_custom(self):
        # Simulate Italy used custom NPIs for a period: each NPI set to 1 for 7 consecutive days
        scenario = [ONE_NPIS] * 7
        self._check_counterfactual(scenario, scenario[0])

    def _check_counterfactual(self, scenario, scenario_npis):
        # Simulate Italy lifted all NPI for this period
        start_date_str = "2020-03-20"
        end_date_str = "2020-03-26"
        countries = ["Italy"]
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)
        self.assertIsNotNone(scenario_df)
        # Misleading name but checks the elements, regardless of order
        self.assertCountEqual(countries, scenario_df.CountryName.unique(), "Not the requested countries")
        self.assertFalse(scenario_df["Date"].duplicated().any(), "Expected 1 row per date only")
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        before_day = start_date - np.timedelta64(1, 'D')
        before_day_npis = scenario_df[scenario_df.Date == before_day][NPI_COLUMNS].reset_index(drop=True)
        before_day_npis_truth = self.latest_df[(self.latest_df.CountryName == "Italy") &
                                               (self.latest_df.Date == before_day)][NPI_COLUMNS].reset_index(drop=True)
        # Check the day before the scenario is correct
        pd.testing.assert_frame_equal(before_day_npis_truth, before_day_npis, "Not the expected frozen NPIs")
        # For the right period (+1 to include start and end date)
        nb_days = (end_date - start_date).days + 1
        for i in range(nb_days):
            check_day = start_date + np.timedelta64(i, 'D')
            check_day_npis_df = scenario_df[scenario_df.Date == check_day][NPI_COLUMNS].reset_index(drop=True)
            check_day_npis = list(check_day_npis_df.values[0])
            self.assertListEqual(scenario_npis, check_day_npis)
        # Check Mar 27 is different from frozen day
        after_day = end_date + np.timedelta64(1, 'D')
        after_day_npis_df = scenario_df[scenario_df.Date == after_day][NPI_COLUMNS].reset_index(drop=True)
        self.assertTrue((scenario_npis - after_day_npis_df.values[0]).any(),
                        "Expected NPIs to be different")
        # Check 27 is indeed equal to truth
        after_day_npis_truth = self.latest_df[(self.latest_df.CountryName == "Italy") &
                                              (self.latest_df.Date == after_day)
                                              ][NPI_COLUMNS].reset_index(drop=True)
        pd.testing.assert_frame_equal(after_day_npis_truth, after_day_npis_df, "Not the expected unfrozen NPIs")

    def test_generate_scenario_future_freeze(self):
        # Simulate Italy froze it's NPIS for the second part of the year
        countries = ["Italy"]
        start_date_str = "2020-07-01"
        end_date_str = "2020-12-31"
        scenario = "Freeze"

        before_day = pd.to_datetime("2020-06-30", format='%Y-%m-%d')
        frozen_npis_df = self.latest_df[(self.latest_df.CountryName == "Italy") &
                                        (self.latest_df.Date == before_day)][NPI_COLUMNS].reset_index(drop=True)
        scenario_npis = list(frozen_npis_df.values[0])

        # Generate the scenario
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check it
        self._check_future(start_date_str=start_date_str,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df[scenario_df.CountryName == countries[0]],
                           scenario_npis=scenario_npis,
                           country=countries[0])

    def test_generate_scenario_future_min(self):
        # Simulate Italy lifted all NPIs for a period
        countries = ["Italy"]
        start_date_str = "2020-07-01"
        end_date_str = "2020-12-31"
        scenario = "MIN"
        scenario_npis = MIN_NPIS

        # Generate the scenario
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check it
        self._check_future(start_date_str=start_date_str,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df[scenario_df.CountryName == countries[0]],
                           scenario_npis=scenario_npis,
                           country=countries[0])

    def test_generate_scenario_future_max(self):
        # Simulate Italy maxed out all NPIs for a period
        countries = ["Italy"]
        start_date_str = "2020-07-01"
        end_date_str = "2020-12-31"
        scenario = "MAX"
        scenario_npis = MAX_NPIS

        # Generate the scenario
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check it
        self._check_future(start_date_str=start_date_str,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df[scenario_df.CountryName == countries[0]],
                           scenario_npis=scenario_npis,
                           country=countries[0])

    def test_generate_scenario_future_custom(self):
        # Simulate Italy used custom NPIs for a period: each NPI set to 1 for 7 consecutive days
        countries = ["Italy"]
        start_date_str = "2020-07-01"
        end_date_str = "2020-12-31"
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        nb_days = (end_date - start_date).days + 1  # +1 to include start date
        scenario = [ONE_NPIS] * nb_days

        # Generate the scenario
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)
        # Check it
        self._check_future(start_date_str=start_date_str,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df[scenario_df.CountryName == countries[0]],
                           scenario_npis=scenario[0],
                           country=countries[0])

    def test_generate_scenario_future_from_last_known_date_freeze(self):
        # Simulate Italy freezes NPIS for the rest of the year
        countries = ["Italy"]
        start_date_str = None
        end_date_str = "2020-12-31"
        scenario = "Freeze"
        last_known_date = self.latest_df[self.latest_df.CountryName == "Italy"].Date.max()
        frozen_npis_df = self.latest_df[(self.latest_df.CountryName == "Italy") &
                                        (self.latest_df.Date == last_known_date)][NPI_COLUMNS].reset_index(drop=True)
        scenario_npis = list(frozen_npis_df.values[0])

        # Generate the scenario
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check it
        self._check_future(start_date_str=start_date_str,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df[scenario_df.CountryName == countries[0]],
                           scenario_npis=scenario_npis,
                           country=countries[0])

    def test_generate_scenario_future_from_last_known_date_min(self):
        # Simulate Italy lifts all NPIs for the rest of the year
        countries = ["Italy"]
        start_date_str = None
        end_date_str = "2020-12-31"
        scenario = "MIN"
        scenario_npis = MIN_NPIS

        # Generate the scenario
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check it
        self._check_future(start_date_str=start_date_str,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df[scenario_df.CountryName == countries[0]],
                           scenario_npis=scenario_npis,
                           country=countries[0])

    def test_generate_scenario_future_from_last_known_date_max(self):
        # Simulate Italy maxes out NPIs for the rest of the year
        countries = ["Italy"]
        start_date_str = None
        end_date_str = "2020-12-31"
        scenario = "MAX"
        scenario_npis = MAX_NPIS

        # Generate the scenario
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check it
        self._check_future(start_date_str=start_date_str,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df[scenario_df.CountryName == countries[0]],
                           scenario_npis=scenario_npis,
                           country=countries[0])

    def test_generate_scenario_future_from_last_known_date_custom(self):
        # Simulate Italy uses custom NPIs for the rest of the year
        countries = ["Italy"]
        last_known_date = self.latest_df[self.latest_df.CountryName == "Italy"].Date.max()
        start_date = last_known_date + np.timedelta64(1, 'D')
        end_date_str = "2020-12-31"
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        nb_days = (end_date - start_date).days + 1  # +1 to include start date
        scenario = [ONE_NPIS] * nb_days

        # Generate the scenario
        scenario_df = generate_scenario(None, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check it
        self._check_future(start_date_str=None,
                           end_date_str=end_date_str,
                           scenario_df=scenario_df,
                           scenario_npis=scenario[0],
                           country=countries[0])

    def test_generate_scenario_all_countries_future_from_last_known_date_freeze(self):
        # Simulate ALL countries uses custom NPIs for the rest of the year
        countries = None
        end_date_str = "2020-12-31"
        # Make sure we generate scenarios for enough days
        nb_days = 180
        scenario = [ONE_NPIS] * nb_days

        # Generate the scenarios
        scenario_df = generate_scenario(None, end_date_str, self.latest_df, countries, scenario=scenario)

        # Check them
        all_countries = self.latest_df.CountryName.unique()
        for country in all_countries:
            all_regions = self.latest_df[self.latest_df.CountryName == country].RegionName.unique()
            for region in all_regions:
                self._check_future(start_date_str=None,
                                   end_date_str=end_date_str,
                                   scenario_df=scenario_df[(scenario_df.CountryName == country) &
                                                           (scenario_df.RegionName == region)],
                                   scenario_npis=scenario[0],
                                   country=country,
                                   region=region)

    def _check_future(self, start_date_str, end_date_str, scenario_df, scenario_npis, country, region=""):
        self.assertIsNotNone(scenario_df)
        # Misleading name but checks the elements, regardless of order
        self.assertCountEqual([country], scenario_df.CountryName.unique(), "Not the expected country")
        self.assertCountEqual([region], scenario_df.RegionName.unique(), "Not the requested region")
        self.assertFalse(scenario_df["Date"].duplicated().any(), "Did not expect duplicated days")
        self.assertFalse(scenario_df["Date"].duplicated().any(), "Expected 1 row per date only")
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        # Check the "historical" period
        last_known_date = self.latest_df[(self.latest_df.CountryName == country) &
                                         (self.latest_df.RegionName == region)].Date.max()
        # If the start date is not specified, start from the day after the last known day
        if not start_date_str:
            start_date = last_known_date + np.timedelta64(1, 'D')
        else:
            start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        past_df = scenario_df[scenario_df.Date < start_date][NPI_COLUMNS].reset_index(drop=True)
        historical_df = self.latest_df[(self.latest_df.CountryName == country) &
                                       (self.latest_df.RegionName == region) &
                                       (self.latest_df.Date < start_date)][NPI_COLUMNS].reset_index(drop=True)
        pd.testing.assert_frame_equal(historical_df, past_df, "Not the expected past NPIs")

        # Check the "future" period (+1 to include start and end date)
        nb_days = (end_date - start_date).days + 1
        for i in range(nb_days):
            check_day = start_date + np.timedelta64(i, 'D')
            self.assertFalse(scenario_df[scenario_df.Date == check_day].empty,
                             f"Expected npis for country {country}, region {region}, day {check_day}")
            check_day_npis_df = scenario_df[scenario_df.Date == check_day][NPI_COLUMNS].reset_index(drop=True)
            check_day_npis = list(check_day_npis_df.values[0])
            self.assertListEqual(scenario_npis, check_day_npis)

        # Check last day is indeed end_date
        self.assertEqual(end_date, scenario_df.tail(1).Date.values[0], "Not the expected end date")

    def test_generate_scenario_mind_the_gap_freeze(self):
        # Scenario = Freeze
        start_date_str = "2021-01-01"
        end_date_str = "2021-01-31"
        countries = ["Italy"]
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario="Freeze")
        self.assertIsNotNone(scenario_df)
        # Misleading name but checks the elements, regardless of order
        self.assertCountEqual(countries, scenario_df.CountryName.unique(), "Not the requested countries")
        # Inception is 2020-01-01. 366 days for 2020 + 31 for Jan 2021
        self.assertEqual(397, len(scenario_df), "Expected the number of days between inception and end date")
        # The last 31 rows must be the same
        self.assertEqual(0, scenario_df.tail(31)[NPI_COLUMNS].diff().sum().sum(),
                         "Expected the last 31 rows to have the same frozen IP")

    def test_generate_scenario_mind_the_gap_min(self):
        # Scenario = MIN
        start_date_str = "2021-01-01"
        end_date_str = "2021-01-31"
        countries = ["Italy"]
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario="MIN")
        self.assertIsNotNone(scenario_df)
        # Misleading name but checks the elements, regardless of order
        self.assertCountEqual(countries, scenario_df.CountryName.unique(), "Not the requested countries")
        # Inception is 2020-01-01. 366 days for 2020 + 31 for Jan 2021
        self.assertEqual(397, len(scenario_df), "Expected the number of days between inception and end date")
        # The last 31 rows must be the same
        self.assertEqual(0, scenario_df.tail(31)[NPI_COLUMNS].sum().sum(),
                         "Expected the last 31 rows to have NPIs set to 0")

    def test_generate_scenario_mind_the_gap_max(self):
        # Scenario = MAX
        start_date_str = "2021-01-01"
        end_date_str = "2021-01-31"
        countries = ["Italy"]
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario="MAX")
        self.assertIsNotNone(scenario_df)
        # Misleading name but checks the elements, regardless of order
        self.assertCountEqual(countries, scenario_df.CountryName.unique(), "Not the requested countries")
        # Inception is 2020-01-01. 366 days for 2020 + 31 for Jan 2021
        self.assertEqual(397, len(scenario_df), "Expected the number of days between inception and end date")
        # The last 31 rows must be the same
        self.assertEqual(sum(MAX_NPIS), scenario_df.tail(31)[NPI_COLUMNS].mean().sum(),
                         "Expected the last 31 rows to have NPIs set to their max value")

    def test_generate_scenario_mind_the_gap_custom(self):
        # Scenario = Custom
        start_date_str = "2021-01-01"
        end_date_str = "2021-01-31"
        countries = ["Italy"]
        # Set all the NPIs to one for each day between start data and end date.
        scenario = [ONE_NPIS] * 31
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario=scenario)
        self.assertIsNotNone(scenario_df)
        # Misleading name but checks the elements, regardless of order
        self.assertCountEqual(countries, scenario_df.CountryName.unique(), "Not the requested countries")
        # Inception is 2020-01-01. 366 days for 2020 + 31 for Jan 2021
        self.assertEqual(397, len(scenario_df), "Expected the number of days between inception and end date")
        # The last 31 rows must be the same
        self.assertEqual(1, scenario_df.tail(31)[NPI_COLUMNS].mean().mean(),
                         "Expected the last 31 rows to have all NPIs set to 1")

    def test_generate_scenario_mind_the_gap_freeze_2_countries(self):
        # Check 2 countries
        start_date_str = "2021-01-01"
        end_date_str = "2021-01-31"
        countries = ["France", "Italy"]
        scenario_df = generate_scenario(start_date_str, end_date_str, self.latest_df, countries, scenario="Freeze")
        self.assertIsNotNone(scenario_df)
        # Misleading name but checks the elements, regardless of order
        self.assertCountEqual(countries, scenario_df.CountryName.unique(), "Not the requested countries")
        # Inception is 2020-01-01. 366 days for 2020 + 31 for Jan 2021
        self.assertEqual(397 * 2, len(scenario_df), "Expected the number of days between inception and end date")

