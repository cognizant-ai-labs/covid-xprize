# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest

from covid_xprize.validation.predictor_validation import validate_submission, PREDICTED_DAILY_NEW_CASES

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, "fixtures")
PREDS_PATH = os.path.join(FIXTURES_PATH, "predictor")

IP_FILE_ALL_COUNTRIES = os.path.join(FIXTURES_PATH, "ip_file_all_countries.csv")
IP_FILE_FEW_COUNTRIES = os.path.join(FIXTURES_PATH, "ip_file_few_countries.csv")
WRONG_COLUMNS = os.path.join(PREDS_PATH, "wrong_columns.csv")
VALID_SUBMISSION = os.path.join(PREDS_PATH, "valid_submission.csv")
VALID_WITH_ADD_COLS_SUBMISSION = os.path.join(PREDS_PATH, "valid_with_add_cols_submission.csv")
NAN_SUBMISSION = os.path.join(PREDS_PATH, "nan_submission.csv")
NEGATIVE_SUBMISSION = os.path.join(PREDS_PATH, "negative_submission.csv")
MISSING_COUNTRY_SUBMISSION = os.path.join(PREDS_PATH, "missing_country_submission.csv")
BAD_DATES_SUBMISSION = os.path.join(PREDS_PATH, "bad_dates_submission.csv")


class TestPredictionValidation(unittest.TestCase):

    def test_wrong_columns(self):
        errors = validate_submission("2020-08-01", "2020-08-01", IP_FILE_ALL_COUNTRIES, WRONG_COLUMNS)
        self.assertIsNotNone(errors)
        self.assertTrue(PREDICTED_DAILY_NEW_CASES in errors[0])

    def test_valid_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", IP_FILE_ALL_COUNTRIES, VALID_SUBMISSION)
        self.assertTrue(not errors, f"Unexpected errors: {errors}")

    def test_valid_with_additional_columns_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", IP_FILE_FEW_COUNTRIES, VALID_WITH_ADD_COLS_SUBMISSION)
        self.assertTrue(not errors, f"Unexpected errors: {errors}")

    def test_nan_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", IP_FILE_FEW_COUNTRIES, NAN_SUBMISSION)
        self.assertIsNotNone(errors)
        self.assertTrue("NaN" in errors[0], f"Expected 'NaN' in errors, but got {errors}")

    def test_negative_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", IP_FILE_FEW_COUNTRIES, NEGATIVE_SUBMISSION)
        self.assertIsNotNone(errors)
        self.assertTrue("negative" in errors[0], f"Expected 'negative' in errors, but got {errors}")

    def test_missing_country_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", IP_FILE_FEW_COUNTRIES, MISSING_COUNTRY_SUBMISSION)
        self.assertIsNotNone(errors)
        self.assertTrue("Aruba" in errors[0], f"Expected 'Aruba' in errors, because it's missing, but got {errors}")

    def test_dates(self):
        errors = validate_submission("2020-08-01", "2020-08-04", IP_FILE_FEW_COUNTRIES, BAD_DATES_SUBMISSION)
        self.assertIsNotNone(errors)
        self.assertEqual(10, len(errors), "Not the expected number of errors")
        expected_errors = ['Afghanistan: Expected prediction for date 2020-08-01 but got 2020-07-01',
                           'Afghanistan: Expected prediction for date 2020-08-02 but got 2020-07-02',
                           'Afghanistan: Expected prediction for date 2020-08-03 but got 2020-07-03',
                           'Afghanistan: Expected prediction for date 2020-08-04 but got 2020-07-04',
                           'Albania: Expected prediction for date 2020-08-03 but got None',
                           'Albania: Expected prediction for date 2020-08-04 but got None',
                           'Angola: Expected prediction for date None but got 2020-08-05',
                           'Aruba: Expected prediction for date 2020-08-02 but got 2020-08-03',
                           'Aruba: Expected prediction for date 2020-08-03 but got 2020-08-04',
                           'Aruba: Expected prediction for date 2020-08-04 but got None']
        self.assertEqual(expected_errors, errors)

