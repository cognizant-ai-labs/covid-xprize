# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import unittest

from validation.prescriptor_validation import validate_submission


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, "fixtures")
PRESCRIPTIONS_PATH = os.path.join(FIXTURES_PATH, "prescriptor")

IP_FILE_ALL_COUNTRIES = os.path.join(FIXTURES_PATH, "ip_file_all_countries.csv")
IP_FILE_FEW_COUNTRIES = os.path.join(FIXTURES_PATH, "ip_file_few_countries.csv")
WRONG_COLUMNS = os.path.join(PRESCRIPTIONS_PATH, "wrong_columns.csv")
VALID_SUBMISSION = os.path.join(PRESCRIPTIONS_PATH, "valid_submission.csv")

MISSING_COLUMNS = ["C1_School closing", "PrescriptionIndex"]


class TestPrescriptionValidation(unittest.TestCase):

    def test_wrong_columns(self):
        errors = validate_submission("2020-08-01", "2020-08-01", IP_FILE_ALL_COUNTRIES, WRONG_COLUMNS)
        self.assertIsNotNone(errors)
        self.assertTrue(MISSING_COLUMNS[0] in errors[0])
        self.assertTrue(MISSING_COLUMNS[1] in errors[0])

    def test_valid_submission(self):
        errors = validate_submission("2020-08-01", "2020-08-04", IP_FILE_ALL_COUNTRIES, VALID_SUBMISSION)
        self.assertTrue(not errors, f"Unexpected errors: {errors}")
