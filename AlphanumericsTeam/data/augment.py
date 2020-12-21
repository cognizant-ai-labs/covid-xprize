import os, sys

# add the parent directory to see the other atc library dependencies
# NOTE: the change to sys.path is not needed if the virtual env has been
# installed using script run.sh at root directory of repo
FILE_PATH = os.getcwd()
sys.path.append(FILE_PATH + '//..//..//')
# print(sys.path) # echo the system path

from AlphanumericsTeam.data.util import VALID_COUNTRIES, VALID_REGIONS, get_orig_oxford_df, get_pop_df, DATA_FILE_PATH
from AlphanumericsTeam.data.holiday import holiday_area
from pprint import pprint
from tqdm import tqdm
import pandas as pd


oxford_df = get_orig_oxford_df()


def applyFunc(row, pop_df):

    date = row["Date"].to_pydatetime().strftime("%Y%m%d")

    if row["Jurisdiction"] == "NAT_TOTAL":
        if row["CountryCode"] not in VALID_COUNTRIES:
            res = [None, None, None, None]
        else:
            res = [holiday_area(row["CountryCode"], date)]
            res.extend(pop_df[pop_df["Code"]==row["CountryCode"]].values.tolist()[0][2:5])
        return pd.Series(res)

    if row["Jurisdiction"] == "STATE_TOTAL":
        if row["RegionCode"] not in VALID_REGIONS:
            res = [None, None, None, None]
        else:
            res = [holiday_area(row["RegionCode"], date)]
            res.extend(pop_df[pop_df["Code"]==row["RegionCode"]].values.tolist()[0][2:5])
        return pd.Series(res)


# Create and register a new `tqdm` instance with `pandas`
oxford_df['NewCases'] = oxford_df.ConfirmedCases.diff().fillna(0)
oxford_df['GeoID'] = oxford_df['CountryName'] + '__' + oxford_df['RegionName'].astype(str)

# Get population data
pop_df = get_pop_df()
tqdm.pandas()
oxford_df[['Holidays','pop_2020', 'area_km2','density_perkm2']] = oxford_df.progress_apply(applyFunc, pop_df=pop_df, axis=1)

oxford_df['Date'] = oxford_df['Date'].dt.strftime('%Y%m%d')

AUG_DATA_PATH = os.path.join(DATA_FILE_PATH, "OxCGRT_latest_aug.csv")
oxford_df.to_csv(AUG_DATA_PATH, index = False)