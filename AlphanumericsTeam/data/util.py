import os
from pprint import pprint
import pandas as pd

cd = os.path.dirname(os.path.realpath(__file__))

ROOT_DIRECTORY = os.path.abspath(os.path.join(cd, os.pardir))
DATA_FILE_PATH = os.path.join(ROOT_DIRECTORY, "data", "files")

FINAL_COUNTRIES = dict(
            {}
)
print(ROOT_DIRECTORY)
print(DATA_FILE_PATH)

def get_country_codes():
    country_code_file  = os.path.join(DATA_FILE_PATH, "country_codes.txt")
    country_codes = set()
    with open(country_code_file) as f:
        for line in f:
            l = line.rstrip().split(",")
            country_codes.add((l[0], l[1] ))

    return country_codes


def get_region_codes():
    region_code_file  = os.path.join(DATA_FILE_PATH, "region_codes.txt")
    region_codes = set()
    with open(region_code_file) as f:
        for line in f:
            l = line.rstrip().split(",")
            region_codes.add((l[0], l[1], l[2], l[3] ))

    return region_codes


def get_orig_df():
    DATA_URL = os.path.join(DATA_FILE_PATH, "OxCGRT_latest.csv")
    oxford_df = pd.read_csv(DATA_URL,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                    "RegionCode": str},
                            error_bad_lines=False)

    oxford_df["RegionName"] = oxford_df["RegionName"].fillna("")
    return oxford_df


def get_aug_df():
    DATA_URL = os.path.join(DATA_FILE_PATH, "OxCGRT_latest_aug.csv")
    oxford_df = pd.read_csv(DATA_URL,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                    "RegionCode": str},
                            error_bad_lines=False)

    oxford_df["RegionName"] = oxford_df["RegionName"].fillna("")
    return oxford_df