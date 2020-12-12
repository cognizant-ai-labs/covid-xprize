import sys
import pandas as pd
from sklearn import metrics
import sys

DATA_FILE = 'covid_xprize/data_sources/OxCGRT_latest.csv'
df = pd.read_csv(DATA_FILE, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)


# Add RegionID column that combines CountryName and RegionName for easier manipulation of data
df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)

# Add new cases column
df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

# Keep only columns of interest
id_cols = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
cases_col = ['NewCases']
df = df[id_cols + cases_col]

# Fill any missing case values by interpolation and setting NaNs to 0
df.update(df.groupby('GeoID').NewCases.apply(
    lambda group: group.interpolate()).fillna(0))

# output = "predictions/2020-11-13_2020-12-05.csv"
output = sys.argv[1]
output = pd.read_csv(output, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
# Add RegionID column that combines CountryName and RegionName for easier manipulation of data
output['GeoID'] = output['CountryName'] + '__' + output['RegionName'].astype(str)

res = pd.merge(df, output, how="inner")
y = res.NewCases.rolling(7, min_periods=1).mean()
hy = res.PredictedDailyNewCases.rolling(7, min_periods=1).mean()
print("R2: %0.4f" % metrics.r2_score(y, hy))
print("MAE: %0.4f" % metrics.mean_absolute_error(y, hy))

