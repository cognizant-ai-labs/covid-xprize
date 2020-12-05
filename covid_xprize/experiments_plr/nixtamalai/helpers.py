import pandas as pd

def add_test_data(oxford_path, tests_path):
    """Returns a dataframe with Oxford data merged with covid tests data.
    
    The returned dataframe contains the same number of rows as the oxford dataset
    Covid tests data comes from:
    https://github.com/owid/covid-19-data/blob/master/public/data/testing/covid-testing-all-observations.csv
    
    Parameters:
    oxford_path (str): path to oxford dataset
    tests_path (str): path to tests_daaset
    
    """
    covid_tests = pd.read_csv(tests_path, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
    # drop rows with null Code
    covid_tests = covid_tests[covid_tests.Code.notna()]
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
                   .drop(['Entity', 'new_tests_per_thousand_7day_smoothed Annotations'], axis=1)
                   .rename({'new_tests_per_thousand_7day_smoothed': 'covid_tests'})
                  )
    return oxford_tests.reset_index()