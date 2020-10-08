import numpy as np
import pandas as pd

from common.constants import Constants


def load_dataset(url: str = Constants.LATEST_DATA_URL) -> pd.DataFrame:
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
