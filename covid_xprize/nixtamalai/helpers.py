import pandas as pd
import numpy as np
import urllib
from os import path

CUR_DIRECTORY_PATH = path.abspath(path.dirname(__file__))
DATA_PATH = path.join(path.split(CUR_DIRECTORY_PATH)[0], 'data_sources')
ADDITIONAL_CONTEXT_FILE = path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = path.join(DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = path.join(DATA_PATH, "brazil_populations.csv")
COUNTRIES_REGIONS = path.join(DATA_PATH, "countries_regions.csv")
IMF_REGIONS = path.join(DATA_PATH, "imf_economic_regions.csv")
US_PREFIX = "United States / "
NPI_COLS = ['C1_School closing',
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
STATIC_COLS =['economic_freedom',
              'gni_per_capita', 
              'human_development',
              'life_expectancy',
              'average_years_school',
              'political_regime',
              'Population',
              'pop_density',
              'urban_pop',
              '65_plus',
              'obesity_rate',
              'cancer_rate',
              'smoking_deaths',
              'pneumonia_deaths',
              'air_pullution_deaths',
              'co2_emissions',
              'airplane_departures']
ID_COLS = ['CountryName',
           'RegionName',
           'CountryCode',
           'GeoID',
           'Date']
CASES_COL = ['NewCasesHampel']            


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


def get_OxCGRT():
    _ = path.join(path.split(CUR_DIRECTORY_PATH)[0], 'data_sources')
    OXFORD_FILE = path.join(_, 'OxCGRT_latest.csv')
    return pd.read_csv(OXFORD_FILE, 
                       parse_dates=['Date'],
                       encoding="ISO-8859-1",
                       dtype={"RegionName": str,
                              "RegionCode": str},
                       error_bad_lines=False)        


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


def preprocess_npi(df: pd.DataFrame):
    # Add GeoID
    df['GeoID'] = np.where(df["RegionName"].isnull(),
                                      df["CountryName"],
                                      df["CountryName"] + ' / ' + df["RegionName"])
    # Missing data in NPIs assuming they are the same as previous day
    for npi_col in NPI_COLS:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))
    return df


def preprocess_newcases(df: pd.DataFrame):
    # Add NewCases
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    # Missing data in NewCases
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))
    return df

def get_additional_context() -> pd.DataFrame:
    rename_cols = {'Population Density (# per km2)': 'pop_density',
                    'Urban population (% of total population)': 'urban_pop',
                    'Population ages 65 and above (% of total population)': '65_plus',
                    'Obesity Rate (%)': 'obesity_rate',
                    'Cancer Rate (%)': 'cancer_rate',
                    'Share of Deaths from Smoking (%)': 'smoking_deaths',
                    'Pneumonia Death Rate (per 100K)': 'pneumonia_deaths',
                    'Share of Deaths from Air Pollution (%)': 'air_pullution_deaths',
                    'CO2 emissions (metric tons per capita)': 'co2_emissions',
                    'Air transport (# carrier departures worldwide)': 'airplane_departures'
                    }
    additional_data_df = (pd.read_csv(ADDITIONAL_CONTEXT_FILE)
                          .drop('GDP per capita (current US$)', axis=1)
                         )
    additional_data_df['GeoID'] = additional_data_df['CountryName']
    additional_data_df = additional_data_df.rename(rename_cols, axis=1)
    # Impute variables with imf regions 
    regions = pd.read_csv(IMF_REGIONS).drop('Unnamed: 0', axis=1)
    additional_data_df = additional_data_df.merge(regions[['CountryCode', 'imf_region']], 
                                                  on='CountryCode')
    additional_data_df = (additional_data_df
                          .groupby('imf_region')
                          .apply(lambda group: group.fillna(group.mean()))
                          .drop('imf_region', axis=1)
                          .reset_index()
                          .drop(['level_1', 'imf_region'], axis=1)
                         )
    return additional_data_df



def preprocess_full(k=7, threshold=3, merge_owd='imputed', tests=False):
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
    k: size of window (including the sample; 7 is equal to 3 on either side of value).
    threshold: number of standard deviations to filter outliers
    merge_owd: 'imputed' -> merges imputed data; 
               'original' -> merges original data;
               anything else -> don't merge OWD  
    
    Returns
    Dataframe with all variables merged and preprocessed.
    """
    # get updated data merged with tests
    if tests:
        df = get_OxCGRT_tests()
        tests_columns = [c for c in df.columns if c.startswith('tests')]
        all_columns = ID_COLS + CASES_COL + NPI_COLS + tests_columns
        # Missing data in Tests
        for column in tests_columns:
            df.update(df.groupby('GeoID')[column].apply(
            lambda group: group.interpolate()).fillna(0))
    else:
        all_columns = ID_COLS + CASES_COL + NPI_COLS
        df = get_OxCGRT()

    df = (df.pipe(preprocess_npi)
            .pipe(preprocess_newcases)
    )
    # Hampel filter (default values)
    filtered = df.groupby('CountryCode').apply(lambda group: hampel(group.NewCases, k, threshold))
    filtered = filtered.reset_index()[['NewCases']]
    filtered.columns = ['NewCasesHampel']
    df = df.join(filtered)
    df = df[all_columns]
    if merge_owd == 'imputed':
        _ = path.join(DATA_PATH, 'owd_by_country_imputed.csv')
        owd = pd.read_csv(_).drop('Unnamed: 0', axis=1)
        df = (df.merge(owd, on='CountryCode', how='left')
              .drop('imf_region', axis=1))
    elif merge_owd == 'original':
        _ = path.join(DATA_PATH, 'owd_by_country.csv')
        owd = pd.read_csv(_).drop('Unnamed: 0', axis=1)
        df = df.merge(owd, on='CountryCode', how='left')
    # Merge additional data from xprize
    additional_data = get_additional_context()
    # US state level population
    us_states_pop_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                                usecols=['NAME', 'POPESTIMATE2019'])
    # change names so we can simply append the dataframes
    us_states_pop_df = us_states_pop_df.rename(columns={'POPESTIMATE2019': 'Population'})
    # GeoID for the states
    us_states_pop_df['GeoID'] = US_PREFIX + us_states_pop_df['NAME']
    # Append
    additional_data = additional_data.append(us_states_pop_df)
    # UK population 
    uk_pop_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
    # Append
    additional_data = additional_data.append(uk_pop_df)
    # Brazil population
    brazil_pop_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
    additional_data = (additional_data.append(brazil_pop_df)
                      .drop('NAME', axis=1)
                   )
    df = (df.merge(additional_data, on='CountryName', how='left')
          .drop('GeoID_y', axis=1)
          .rename({'GeoID_x': 'GeoID'}, axis=1)
          .dropna(subset=['Population'])
          )
    # Filter countries not used for evaluation
    cr = pd.read_csv(COUNTRIES_REGIONS)
    cr['GeoID'] = np.where(cr["RegionName"].isnull(),
                                    cr["CountryName"],
                                    cr["CountryName"] + ' / ' + cr["RegionName"])
    rois = cr.GeoID.unique()
    df = df[df.GeoID.isin(rois)]
    return df

def mae(pred, true):
    """Return the Median Absolute Error.
    
    Parameters
    pred: array with predicted values
    true: array with ground truth
    """
    return np.mean(np.abs(pred - true))
    