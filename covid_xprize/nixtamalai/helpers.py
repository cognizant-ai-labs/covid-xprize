import pandas as pd

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