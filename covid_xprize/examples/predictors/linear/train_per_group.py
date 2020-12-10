import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import urllib.request
from sklearn.linear_model import LarsCV


# Main source for the training data
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
# Local file
DATA_FILE = 'data/OxCGRT_latest.csv'

if not os.path.exists('data'):
    os.mkdir('data')
urllib.request.urlretrieve(DATA_URL, DATA_FILE)

# Load historical data from local file
df = pd.read_csv(DATA_FILE, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)

# For testing, restrict training data to that before a hypothetical predictor submission date
# vamos a predecir del 13 de nov al 5 de diciembre para predecir 23 días
# como es en la competencia
HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-11-12")
df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]
df =  df.loc[df.ConfirmedCases >= 32]                 

# Add RegionID column that combines CountryName and RegionName for easier manipulation of data
df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)

# Add new cases column
df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0).rolling(7, min_periods=1).mean().apply(np.log).replace([np.inf, -np.inf], np.nan).fillna(0)

# Keep only columns of interest
id_cols = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
cases_col = ['NewCases']
npi_cols = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']
df = df[id_cols + cases_col + npi_cols]

# Fill any missing case values by interpolation and setting NaNs to 0
df.update(df.groupby('GeoID').NewCases.apply(
    lambda group: group.interpolate()).fillna(0))

# Fill any missing NPIs by assuming they are the same as previous day
for npi_col in npi_cols:
    df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))


MODELS = dict()
LAGS = 4

for key, value in tqdm(df.groupby("GeoID")):
    X = value.loc[:, npi_cols + cases_col].to_numpy()
    features = list()
    y = list()
    for lag in range(LAGS, X.shape[0] - LAGS - 1):
        features.append(X[lag - LAGS:lag].flatten())
        y.append(X[lag, -1])
    m = LarsCV().fit(np.array(features), np.array(y))
    MODELS[key] = m

# Save model to file
if not os.path.exists('models'):
    os.mkdir('models')
with open('models/lars.pkl', 'wb') as model_file:
    pickle.dump(MODELS, model_file)