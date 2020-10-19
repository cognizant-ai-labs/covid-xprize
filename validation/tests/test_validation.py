# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest

from validation.validation import validate_submission, PREDICTED_DAILY_NEW_CASES

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
WRONG_COLUMNS = os.path.join(FIXTURES_PATH, "wrong_columns.csv")
VALID_SUBMISSION = os.path.join(FIXTURES_PATH, "valid_submission.csv")
NAN_SUBMISSION = os.path.join(FIXTURES_PATH, "nan_submission.csv")
NEGATIVE_SUBMISSION = os.path.join(FIXTURES_PATH, "negative_submission.csv")
BAD_DATES_SUBMISSION = os.path.join(FIXTURES_PATH, "bad_dates_submission.csv")


class TestValidation(unittest.TestCase):

    def test_wrong_columns(self):
        errors = validate_submission("2020-08-01", "2020-08-01", WRONG_COLUMNS)
        self.assertIsNotNone(errors)
        self.assertTrue(PREDICTED_DAILY_NEW_CASES in errors[0])

    def test_valid_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", VALID_SUBMISSION)
        self.assertTrue(not errors)

    def test_nan_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", NAN_SUBMISSION)
        self.assertIsNotNone(errors)
        self.assertTrue("NaN" in errors[0])

    def test_negative_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", NEGATIVE_SUBMISSION)
        self.assertIsNotNone(errors)
        self.assertTrue("negative" in errors[0])

    def test_dates(self):
        errors = validate_submission("2020-08-01", "2020-08-04", BAD_DATES_SUBMISSION)
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
