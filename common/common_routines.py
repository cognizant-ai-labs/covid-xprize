import numpy as np
import pandas as pd

from common.constants import Constants


def load_dataset(url: str = Constants.LATEST_DATA_URL) -> pd.DataFrame:
    latest_df = pd.read_csv(url,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    return latest_df
