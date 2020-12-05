import os
import pandas as pd
import argparse
import datetime
import holidays
from collections import defaultdict
from dateutil.parser import parse
import urllib.request

#from utils.holiday import

LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'

ROOT_DIRECTORY = os.path.abspath(os.path.join(".", os.pardir))
DATA_FILE_PATH = os.path.join(ROOT_DIRECTORY, "data", "files")
CSV_PATH = os.path.join(DATA_FILE_PATH, "OxCGRT_latest.csv")

print("ROOT_DIRECTORY ", ROOT_DIRECTORY)
print("CSV_PATH ", CSV_PATH)

with urllib.request.urlopen(LATEST_DATA_URL) as response, open(CSV_PATH, "wb") as out_file:
    data = response.read() # a `bytes` object
    out_file.write(data)


oxford_df = pd.read_csv(CSV_PATH,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        dtype={"RegionName": str,
                                "RegionCode": str},
                        error_bad_lines=False)

oxford_df["RegionName"] = oxford_df["RegionName"].fillna("")

# get country codes and save to file for reference
country_codes = set([(x,y) for x,y in
                     zip(oxford_df['CountryCode'].to_list(),
                         oxford_df['CountryName'].to_list()
                        )])
country_codes = sorted(list(country_codes))

with open(os.path.join(DATA_FILE_PATH, 'country_codes.txt'), 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in country_codes))

# get region codes and save to file for reference
region_codes = set([(x,y,z,w) for x,y,z,w in
                   zip(oxford_df['RegionCode'].to_list(),
                       oxford_df['CountryCode'].to_list(),
                       oxford_df['CountryName'].to_list(),
                       oxford_df['RegionName'].to_list()) if isinstance(x, str)])
region_codes = sorted(list(region_codes))

with open(os.path.join(DATA_FILE_PATH, 'region_codes.txt'), 'w') as fp:
    fp.write('\n'.join('%s,%s,%s,%s' % x for x in region_codes))
