# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import numpy as np
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
# From https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md
MAX_NPIS = [3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2]  # Sum is 30


def generate_scenario(start_date_str, end_date_str, raw_df, countries=None, scenario="Freeze"):
    """
    Generates a scenario: a list of intervention plans, with history since 1/1/2020.
    By default returns historical data.
    Args:
        start_date_str: start_date from which to apply the scenario
        end_date_str: end_date of the data
        raw_df: the original data frame containing the raw data
        countries: a list of CountryName, or None for all countries
        scenario:
            - "Freeze" to keep the last known IP for every future date
            - "MIN" to set all future IP to 0 (i.e. plan is to take no measures)
            - "MAX" to set all future IP to maximum values (i.e. plan is to do everything possible)
            - an array of integers of NPI_COLUMNS lengths: uses this array as the IP to use.

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
    # Remove any date that is after the requested end_date
    ips_df = ips_df[ips_df.Date <= end_date]

    # Fill any missing NPIs by assuming they are the same as previous day, or 0 if none is available
    for npi_col in NPI_COLUMNS:
        ips_df.update(ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

    future_rows = []
    # Make up IP for dates in the future
    for g in ips_df.CountryName.unique():
        ips_gdf = ips_df[ips_df.CountryName == g]
        last_known_date = ips_gdf.Date.max()
        if scenario == "MIN":
            zero_npis = [0] * len(NPI_COLUMNS)
            future_row_values = list(ips_gdf[ips_gdf.Date == last_known_date][ID_COLS].values[0]) + zero_npis
        elif scenario == "MAX":
            future_row_values = list(ips_gdf[ips_gdf.Date == last_known_date][ID_COLS].values[0]) + MAX_NPIS
        elif scenario == "Freeze":
            future_row_values = ips_gdf[ips_gdf.Date == last_known_date].values[0]
        else:
            future_row_values = list(ips_gdf[ips_gdf.Date == last_known_date][ID_COLS].values[0]) + scenario
        current_date = last_known_date + np.timedelta64(1, 'D')
        while current_date <= end_date:
            new_row = future_row_values.copy()
            new_row[ID_COLS.index("Date")] = current_date
            future_rows.append(new_row)
            current_date = current_date + np.timedelta64(1, 'D')
    if future_rows:
        future_rows_df = pd.DataFrame(future_rows, columns=ips_df.columns)
        ips_df = ips_df.append(future_rows_df)
        ips_df.sort_values(by=ID_COLS, inplace=True)

    return ips_df
