import numpy as np
import pandas as pd

LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
LOCAL_DATA_URL = "tests/fixtures/OxCGRT_latest.csv"

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


class XPrizeRobojudge(object):

    def load_dataset(self, url: str = LATEST_DATA_URL) -> pd.DataFrame:
        latest_df = pd.read_csv(url,
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
        return latest_df

    def get_npis(self,
                 start_date: np.datetime64,
                 end_date: np.datetime64,
                 url: str = LATEST_DATA_URL) -> pd.DataFrame:
        latest_df = self.load_dataset(url)
        npis_df = latest_df[["CountryName", "Date"] + NPI_COLUMNS]
        actual_npis_df = npis_df[(npis_df.Date >= start_date) & (npis_df.Date <= end_date)]
        return actual_npis_df

