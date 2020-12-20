import os
import pandas as pd
import argparse
import datetime
import holidays
from collections import defaultdict
from dateutil.parser import parse

LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'


def load_dataset(url):
    latest_df = pd.read_csv(url,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    return latest_df


# get country codes and save to file for reference
oxford_df = load_dataset(LATEST_DATA_URL)
country_codes = set([(x,y) for x,y in
                     zip(oxford_df['CountryCode'].to_list(),
                         oxford_df['CountryName'].to_list()
                        )])
country_codes = sorted(list(country_codes))
with open('country_codes.txt', 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in country_codes))


# get region codes and save to file for reference
region_codes = set([(x,y,z,w) for x,y,z,w in
                   zip(oxford_df['RegionCode'].to_list(),
                       oxford_df['CountryCode'].to_list(),
                       oxford_df['CountryName'].to_list(),
                       oxford_df['RegionName'].to_list()) if isinstance(x, str)])
region_codes = sorted(list(region_codes))

with open('region_codes.txt', 'w') as fp:
    fp.write('\n'.join('%s %s %s %s' % x for x in region_codes))


# Key for weekdays/weekend
#  Mon   Tue   Wed   Thu   Fri   Sat   Sun
#   0     1     2     3     4     5     6
WEEKENDS = defaultdict(lambda : (5,6))
WEEKENDS.update({
    "AFG": (3,4),   # Afghanistan
    "DZA": (4,5),   # Algeria
    "BHR": (4,5),   # Bahrain
    "BGD": (4,5),   # Bangladesh
    "BRN": (4,6),   # Brunei
    "DJI": (None,4),    # Djibouti
    "EGY": (4,5),   # Egypt
    "GIN": (None,6),    # Eq, Guinea
    "HKG": (None,6),    #Hong Kong
    "IRN": (None,4),    # Iran
    "IRQ": (4,5),   # Iraq
    "ISR": (4,5),   # Israel
    "JOR": (4,5),   # Jordan
    "KWT": (4,5),   # Kuwait
    "LBY": (4,5),   # Libya
    "NPL": (None,5),    # Nepal
    "OMN": (4,5),   # Oman
    "PSE": (None,4),    # Palestine
    "QAT": (4,5),   # Qatar
    "SAU": (4,5),   # Saudi Arabia
    "SOM": (None,4),    # Somalia
    "SDN": (4,5),   # Sudan
    "SYR": (4,5),   # Syria
    "ARE": (4,5),   # UAE
    "UGA": (None,6),    # Uganda
    "YEM": (4,5),   # Yemen
    })


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


def holiday_country(country, date):

    # the two anomalies in code in holidays w.r.t oxford country codes
    # case for morocco and bulgaria
    if country == "MAR":
        country == "MOR"
    if country == "BGR":
        country == "BLG"

    if country in holidays.list_supported_countries():
        func = getattr(holidays, country)
        if date in func():
            return 2  # return holiday (priority over weekend)

    return int( parse(date).weekday() in WEEKENDS[country])


def holiday_region(region, date):
    # get the country and region from region code
    country, reg = region.split("_")

    # convert two character code to oxford compatible three character codes
    if country == "BR":
        country == "BRA"
    elif country == "US":
        country == "USA"
    elif country == "UK":
        country == "GBR"

    # get the attribute for the country
    func = getattr(holidays, country)

    if date in func(state=reg):  # holiday for the state
        return 2  # return holiday (priority over weekend)

    return int( parse(date).weekday() in WEEKENDS[country])  # check for weekend


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
        if args.country_code in ["USA", "GBR", "BRA"]:
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
