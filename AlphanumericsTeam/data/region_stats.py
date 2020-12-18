import os, sys
import pandas as pd
import argparse
import datetime
import holidays
from collections import defaultdict
from dateutil.parser import parse
from AlphanumericsTeam.data.util import get_region_codes, get_country_codes, get_orig_df
from AlphanumericsTeam.data.holiday import holiday_country, holiday_region

LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'

country_codes = get_country_codes()
region_codes = get_region_codes()


def valid_country(s):
    try:
        assert s in  set([x for x,_ in country_codes])
        return s
    except (ValueError, AssertionError):
        msg = "Not a valid country code: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def valid_region(s):
    try:
        assert s in  set([x for x,_,_,_ in region_codes])
        return s
    except (ValueError, AssertionError):
        msg = "Not a valid region code: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def valid_date(s):
    try:
        today = datetime.date.today()
        arg =   datetime.datetime.strptime(s, "%Y-%m-%d").date()
        initial = datetime.date(year=2020, month=1, day=1)
        assert initial <= arg <=today
        return arg
    except (ValueError, AssertionError):
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-r', '--region-code', type=valid_region,
                        help="region to select data from")

    group.add_argument('-c', '--country-code', type=valid_country,
                        help="country to select data from")

    parser.add_argument('-s', '--start-date',  required=True, type=valid_date,
                        help="the start date (in yyyy-mm-dd) to begin getting data from")

    parser.add_argument('-e', '--end-date', required=True, type=valid_date,
                        help="the last date (in yyyy-mm-dd) to get data from (note: last date is included)")

    parser.add_argument('-ho', '--holidays', action='store_true',
                        help="request additional column for holidays/weekend")

    args = parser.parse_args()

    start_date = args.start_date.strftime("%Y-%m-%d")
    end_date = args.end_date.strftime("%Y-%m-%d")

    columns_of_interest = ["Date", "ConfirmedCases", "NewCases"]

    oxford_df = get_orig_df()

    if args.region_code:

        country_name, region_name = [(cn, rn) for x, _, cn, rn in region_codes if x == args.region_code][0]

        df_final = oxford_df[(oxford_df["RegionCode"]==args.region_code)]

        # filter between dates
        df_final = df_final[(pd.Timestamp(start_date)<=df_final["Date"])]
        df_final = df_final[(df_final["Date"]<=pd.Timestamp(end_date))]

        # convert date to ordinal like
        df_final['Date'] = df_final['Date'].apply(lambda x: x.to_pydatetime().strftime("%Y%m%d"))

        if args.holidays:
            df_final['Holidays'] = df_final.apply( lambda x: holiday_region(x["RegionCode"], x["Date"]), axis=1)
            columns_of_interest.append("Holidays")
        # Add a column for new cases only
        df_final['NewCases'] = df_final.ConfirmedCases.diff().fillna(0)

        # select columns of interest
        df_final = df_final[columns_of_interest]

        # save to csv
        fname = os.path.join(os.getcwd(),
                             "__".join([country_name,
                                       region_name,
                                       start_date,
                                       end_date]) + ".csv")

        df_final.to_csv(path_or_buf=fname, index=False, na_rep="nan")

    if args.country_code:
        country_name = [y for x, y in country_codes if x == args.country_code][0]
        region_name = "nan"
        df_final = oxford_df[(oxford_df["CountryCode"]==args.country_code)]

        # for three countries with sub-regions look for only the national sum
        if args.country_code in ["USA", "GBR", "BRA", "CAN"]:
            df_final = df_final[(df_final["Jurisdiction"]== "NAT_TOTAL")]

        # filter between dates
        df_final = df_final[(pd.Timestamp(start_date)<=df_final["Date"])]
        df_final = df_final[(df_final["Date"]<=pd.Timestamp(end_date))]

        # convert date to ordinal like
        df_final['Date'] = df_final['Date'].apply(lambda x: x.to_pydatetime().strftime("%Y%m%d"))

        if args.holidays:
            df_final['Holidays'] = df_final.apply( lambda x: holiday_country(x["CountryCode"], x["Date"]), axis=1)
            columns_of_interest.append("Holidays")

        # Add a column for new cases only
        df_final['NewCases'] = df_final.ConfirmedCases.diff().fillna(0)

        # select columns of interest
        df_final = df_final[columns_of_interest]

        # save to csv
        fname = os.path.join(os.getcwd(),
                             "__".join([country_name,
                                       region_name,
                                       start_date,
                                       end_date]) + ".csv")

        df_final.to_csv(path_or_buf=fname, index=False, na_rep="nan")


if __name__ == "__main__":
    main()
