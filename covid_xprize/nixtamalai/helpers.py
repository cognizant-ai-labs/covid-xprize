import pandas as pd
import numpy as np
import urllib

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
    # drop rows with null Code
    covid_tests = covid_tests[covid_tests["Code"].notna()]
    covid_tests = covid_tests.set_index(['Code', 'Date'])
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

def update_OxCGRT_tests():
    """Returns a dataframe with the latest data from oxford and covid tests.
       Fetches latest data from OxCGRT and OWD and merges them
    """
    # source of latest Oxford data
    OXFORD_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
    # source of latest test data
    TESTS_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv"
    # store them locally
    OXFORD_FILE = '../data_sources/OxCGRT_latest.csv'
    TESTS_FILE = '../data_sources/tests_latest.csv'
    urllib.request.urlretrieve(OXFORD_URL, OXFORD_FILE)
    urllib.request.urlretrieve(TESTS_URL, TESTS_FILE)
    return add_test_data(OXFORD_FILE, TESTS_FILE)

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