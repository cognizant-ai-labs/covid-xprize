import pandas as pd
import numpy as np
import os
import urllib
import os
from os import path

CUR_DIRECTORY_PATH = path.abspath(os.path.dirname(__file__))


def add_test_data(oxford_path, tests_path):
    """Returns a dataframe with Oxford data merged with covid tests data.
    
    The returned dataframe contains the same number of rows as the oxford dataset
    Covid tests data comes from:
    https://github.com/owid/covid-19-data/blob/master/public/data/testing/covid-testing-all-observations.csv
    
    Parameters:
    oxford_path (str): path to oxford dataset
    tests_path (str): path to tests_dataset
    
    """
    covid_tests = (pd.read_csv(tests_path, 
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str},
                     error_bad_lines=False)
                    .rename({'ISO code': 'Code'}, axis=1)
                  )
    covid_tests.columns = covid_tests.columns.str.replace(' ', '_')
    covid_tests.columns = ['tests_' + c  if c not in ['Code', 'Date'] 
                           else c for c in covid_tests.columns]
    # drop rows with null Code
    covid_tests = covid_tests[covid_tests["Code"].notna()]
    # set index for merge and drop unnecesary columns
    covid_tests = (covid_tests.set_index(['Code', 'Date'])
                  .drop(['tests_Source_label', 
                         'tests_Source_URL', 
                         'tests_Notes', 
                         'tests_Entity'], axis=1)
                  )
    oxford = pd.read_csv(oxford_path, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
    oxford = oxford.set_index(['CountryCode', 'Date'])
    oxford_tests =(oxford
                   .join(covid_tests.rename_axis(oxford.index.names), how='left')
                  )
    return oxford_tests.reset_index()


def get_OxCGRT_tests():
    _ = path.join(path.split(CUR_DIRECTORY_PATH)[0], 'data_sources')
    OXFORD_FILE = path.join(_, 'OxCGRT_latest.csv')
    TESTS_FILE = path.join(_, 'tests_latest.csv')    
    return add_test_data(OXFORD_FILE, TESTS_FILE)


def update_OxCGRT_tests():
    """Returns a dataframe with the latest data from oxford and covid tests.
       Fetches latest data from OxCGRT and OWD and merges them
    """
    # source of latest Oxford data
    OXFORD_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
    # source of latest test data
    TESTS_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv"
    # store them locally
    _ = path.join(path.split(CUR_DIRECTORY_PATH)[0], 'data_sources')

    OXFORD_FILE = path.join(_, 'OxCGRT_latest.csv')
    TESTS_FILE = path.join(_, 'tests_latest.csv')
    urllib.request.urlretrieve(OXFORD_URL, OXFORD_FILE)
    urllib.request.urlretrieve(TESTS_URL, TESTS_FILE)
    return get_OxCGRT_tests()


def hampel(vals_orig, k=7, threshold=3):
    """Detect and filter outliers in a time series.
    
    Parameters
    vals_orig: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    threshold: number of standard deviations to filter outliers
    
    Returns
    
    """
    
    #Make copy so original not edited
    vals = vals_orig.copy()
    
    #Hampel Filter
    L = 1.4826 # Constant factor to estimate STD from MAD assuming normality
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = threshold * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    
    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''
    
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    return(vals)

def preprocess(k=7, threshold=3, merge_owd='imputed'):
    """Preprocess OxCGRT data.
    - Update data and merge with tests
    - Add CountryID
    - Add NewCases 
    - Handle missing data in NewCases
    - Handle missing data in NPIs
    - Handle missing data in Tests
    - Fix outliers
    - Optionally merge OWD data
    - Return only relevant columns
    
    Parameters
    k: size of window (including the sample; 7 is equal to 3 on either side of value). 0 Do not apply
    threshold: number of standard deviations to filter outliers
    merge_owd: 'imputed' -> merges imputed data; 
               'original' -> merges original data;
               anything else -> don't merge OWD  
    
    Returns
    Dataframe with all variables merged and preprocessed.
    """
    # get updated data merged with tests
    df = get_OxCGRT_tests()
    # Add GeoID
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
    # Add NewCases
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    # Missing data in NewCases
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))
    # Missing data in Tests
    tests_columns = [c for c in df.columns if c.startswith('tests')]
    for column in tests_columns:
        df.update(df.groupby('GeoID')[column].apply(
        lambda group: group.interpolate()).fillna(0))
    # Missing data in NPIs assuming they are the same as previous day
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
    for npi_col in npi_cols:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))
    # Hampel filter (default values)
    if k:
        filtered = df.groupby('CountryCode').apply(lambda group: hampel(group.NewCases, k, threshold))
    filtered = filtered.reset_index()[['NewCases']]
    filtered.columns = ['NewCasesHampel']
    df = df.join(filtered)
    id_cols = ['CountryName',
               'CountryCode',
               'RegionName',
               'GeoID',
               'Date']
    cases_col = ['NewCases', 'NewCasesHampel']
    df = df [id_cols + cases_col + npi_cols + tests_columns]
    if merge_owd == 'imputed':
        _ = path.join(path.split(CUR_DIRECTORY_PATH)[0], 'data_sources')
        _ = path.join(_, 'owd_by_country_imputed.csv')
        owd = pd.read_csv(_).drop('Unnamed: 0', axis=1)
        df = df.merge(owd, on='CountryCode', how='left')
    elif merge_owd == 'original':
        _ = path.join(path.split(CUR_DIRECTORY_PATH)[0], 'data_sources')
        _ = path.join(_, 'owd_by_country.csv')
        owd = pd.read_csv("../data_sources/owd_by_country.csv").drop('Unnamed: 0', axis=1)
        df = df.merge(owd, on='CountryCode', how='left')
    return df

def mae(pred, true):
    """Return the Median Absolute Error.
    
    Parameters
    pred: array with predicted values
    true: array with ground truth
    """
    return np.mean(np.abs(pred - true))
    