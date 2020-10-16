import pandas as pd

ID_COLS = ['CountryName',
           'RegionName',
           'Date']
NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing']


def generate_scenario(start_date_str, end_date_str, raw_df, countries=None, scenario=None):
    """
    Generates a scenario: a list of intervention plans, with history since 1/1/2020.
    By default returns historical data.
    Args:
        start_date_str: start_date from which to apply the scenario
        end_date_str: end_date of the data
        raw_df: the original data frame containing the raw data
        countries: a list of CountryName, or None for all countries
        scenario: None, MIN, MAX

    Returns: a Pandas DataFrame

    """
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    ips_df = raw_df[ID_COLS + NPI_COLUMNS]

    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data\n",
    #     hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)

    # Filter on countries
    if countries:
        ips_df = ips_df[ips_df.CountryName.isin(countries)]

    # Check the dates
    # Remove any future date
    ips_df = ips_df[ips_df.Date <= end_date]

    #     for g in ips_df.CountryName.unique():
    #         ips_gdf = ips_df[ips_df.GeoID == g]
    return ips_df
