import os
from AlphanumericsTeam.data.util import get_orig_df, DATA_FILE_PATH
from AlphanumericsTeam.data.holiday import holiday_area
from pprint import pprint
from tqdm import tqdm

oxford_df = get_orig_df()

def fun(row):
    date = row["Date"].to_pydatetime().strftime("%Y%m%d")
    if row["Jurisdiction"] == "NAT_TOTAL":
        return holiday_area(row["CountryCode"], date)
    if row["Jurisdiction"] == "STATE_TOTAL":
        return holiday_area(row["RegionCode"], date)

# Create and register a new `tqdm` instance with `pandas`
tqdm.pandas()
oxford_df['Holidays'] = oxford_df.progress_apply(fun, axis=1)
oxford_df['NewCases'] = oxford_df.ConfirmedCases.diff().fillna(0)


AUG_DATA_PATH = os.path.join(DATA_FILE_PATH, "OxCGRT_latest_aug.csv")
print(AUG_DATA_PATH)
oxford_df.to_csv(AUG_DATA_PATH, index = False)