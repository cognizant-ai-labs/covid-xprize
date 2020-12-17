import sys
import pandas as pd
import numpy as np
from sklearn import metrics
from covid_xprize.nixtamalai.helpers import ID_COLS, CASES_COL
from covid_xprize.nixtamalai import helpers
import sys

df = helpers.preprocess_full()
population = {k:v for k, v in df.groupby("GeoID").Population.last().items()}

df = df[ID_COLS + CASES_COL]

# output = "predictions/2020-11-13_2020-12-05.csv"
output = sys.argv[1]
output = pd.read_csv(output, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
# Add RegionID column that combines CountryName and RegionName for easier manipulation of data
output['GeoID'] = np.where(output["RegionName"].isnull(),
                           output["CountryName"],
                           output["CountryName"] + ' / ' + output["RegionName"])

res = pd.merge(df, output, how="inner")
y = res.NewCasesHampel.rolling(7, min_periods=1).mean()
hy = res.PredictedDailyNewCases.rolling(7, min_periods=1).mean()
print("R2: %0.4f" % metrics.r2_score(y, hy))
print("MAE: %0.4f" % metrics.mean_absolute_error(y, hy))

_ = [((100000 * value.NewCasesHampel /  population[key]).rolling(7, min_periods=1).mean().to_numpy(),
      (100000 * value.PredictedDailyNewCases /  population[key]).rolling(7, min_periods=1).mean().to_numpy())
     for key, value in res.groupby("GeoID")]

y = np.concatenate([x[0] for x in _])
hy = np.concatenate([x[1] for x in _])
print("Norm. R2: %0.4f" % metrics.r2_score(y, hy))
print("Norm. MAE: %0.4f" % metrics.mean_absolute_error(y, hy))
