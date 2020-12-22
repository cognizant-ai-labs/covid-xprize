
from AlphanumericsTeam.data.util import get_aug_oxford_df
from collections import defaultdict
import holidays
from dateutil.parser import parse
import pandas as pd
import os


ROOT_DIRECTORY = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DATA_FILE_PATH = os.path.join(ROOT_DIRECTORY, "data")
#print(ROOT_DIRECTORY, "root directory")

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
    if country == "US":
        country = "USA"
    elif country == "UK":
        country = "GBR"
    else:
        country = None

    if country:
        # get the attribute for the country
        func = getattr(holidays, country)

        if date in func(state=reg):  # holiday for the state
            return 2  # return holiday (priority over weekend)

    return int( parse(date).weekday() in WEEKENDS[country])  # check for weekend


def holiday_area(area_code, date):
    if "_" in area_code:
        return holiday_region(area_code, date)
    else:
        return holiday_country(area_code, date)


def holiday_areaname(country_name, region_name, date):

    code  = REG_C_MAP[(country_name, region_name)]

    if "_" in code:
        return holiday_region(code, date)
    else:
        return holiday_country(code, date)


def get_pop_df():
    DATA_URL = os.path.join(DATA_FILE_PATH, "pop_t.csv")
    pop_df = pd.read_csv(DATA_URL)
    return pop_df


def applyFunc(row, pop_df):


    date = row["Date"].to_pydatetime().strftime("%Y%m%d")
    # date = row["Date"]
    cname = row["CountryName"]
    rname = row["RegionName"]
    pop_df["RegionName"] = pop_df["RegionName"].fillna(value="")

    print( cname, rname, date , "---")
    # print( pop_df["CountryName"], pop_df["RegionName"] , "---")

    pop = pop_df[(pop_df["CountryName"] == cname) &
                  (pop_df["RegionName"] == rname)
                  ].iloc[0][2]

    res = [holiday_areaname(cname, rname, date), pop]
    return pd.Series(res)


def add_features_df(df):
    # Get population data
    pop_df = get_pop_df()
    df[['Holidays', 'density_perkm2']] = df.apply(applyFunc, pop_df=pop_df, axis=1)
    return df






# Key for weekdays/weekend
#  Mon   Tue   Wed   Thu   Fri   Sat   Sun
#   0     1     2     3     4     5     6
WEEKENDS = defaultdict(lambda : (5, 6))
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


REG_C_MAP = {('Afghanistan', ''): 'AFG',
             ('Albania', ''): 'ALB',
             ('Algeria', ''): 'DZA',
             ('Andorra', ''): 'AND',
             ('Angola', ''): 'AGO',
             ('Argentina', ''): 'ARG',
             ('Aruba', ''): 'ABW',
             ('Australia', ''): 'AUS',
             ('Austria', ''): 'AUT',
             ('Azerbaijan', ''): 'AZE',
             ('Bahamas', ''): 'BHS',
             ('Bahrain', ''): 'BHR',
             ('Bangladesh', ''): 'BGD',
             ('Barbados', ''): 'BRB',
             ('Belarus', ''): 'BLR',
             ('Belgium', ''): 'BEL',
             ('Belize', ''): 'BLZ',
             ('Benin', ''): 'BEN',
             ('Bermuda', ''): 'BMU',
             ('Bhutan', ''): 'BTN',
             ('Bolivia', ''): 'BOL',
             ('Bosnia and Herzegovina', ''): 'BIH',
             ('Botswana', ''): 'BWA',
             ('Brazil', ''): 'BRA',
             ('Brunei', ''): 'BRN',
             ('Bulgaria', ''): 'BGR',
             ('Burkina Faso', ''): 'BFA',
             ('Burundi', ''): 'BDI',
             ('Cambodia', ''): 'KHM',
             ('Cameroon', ''): 'CMR',
             ('Canada', ''): 'CAN',
             ('Cape Verde', ''): 'CPV',
             ('Central African Republic', ''): 'CAF',
             ('Chad', ''): 'TCD',
             ('Chile', ''): 'CHL',
             ('China', ''): 'CHN',
             ('Colombia', ''): 'COL',
             ('Comoros', ''): 'COM',
             ('Congo', ''): 'COG',
             ('Costa Rica', ''): 'CRI',
             ("Cote d'Ivoire", ''): 'CIV',
             ('Croatia', ''): 'HRV',
             ('Cuba', ''): 'CUB',
             ('Cyprus', ''): 'CYP',
             ('Czech Republic', ''): 'CZE',
             ('Democratic Republic of Congo', ''): 'COD',
             ('Denmark', ''): 'DNK',
             ('Djibouti', ''): 'DJI',
             ('Dominica', ''): 'DMA',
             ('Dominican Republic', ''): 'DOM',
             ('Ecuador', ''): 'ECU',
             ('Egypt', ''): 'EGY',
             ('El Salvador', ''): 'SLV',
             ('Eritrea', ''): 'ERI',
             ('Estonia', ''): 'EST',
             ('Eswatini', ''): 'SWZ',
             ('Ethiopia', ''): 'ETH',
             ('Faeroe Islands', ''): 'FRO',
             ('Fiji', ''): 'FJI',
             ('Finland', ''): 'FIN',
             ('France', ''): 'FRA',
             ('Gabon', ''): 'GAB',
             ('Gambia', ''): 'GMB',
             ('Georgia', ''): 'GEO',
             ('Germany', ''): 'DEU',
             ('Ghana', ''): 'GHA',
             ('Greece', ''): 'GRC',
             ('Greenland', ''): 'GRL',
             ('Guam', ''): 'GUM',
             ('Guatemala', ''): 'GTM',
             ('Guinea', ''): 'GIN',
             ('Guyana', ''): 'GUY',
             ('Haiti', ''): 'HTI',
             ('Honduras', ''): 'HND',
             ('Hong Kong', ''): 'HKG',
             ('Hungary', ''): 'HUN',
             ('Iceland', ''): 'ISL',
             ('India', ''): 'IND',
             ('Indonesia', ''): 'IDN',
             ('Iran', ''): 'IRN',
             ('Iraq', ''): 'IRQ',
             ('Ireland', ''): 'IRL',
             ('Israel', ''): 'ISR',
             ('Italy', ''): 'ITA',
             ('Jamaica', ''): 'JAM',
             ('Japan', ''): 'JPN',
             ('Jordan', ''): 'JOR',
             ('Kazakhstan', ''): 'KAZ',
             ('Kenya', ''): 'KEN',
             ('Kosovo', ''): 'RKS',
             ('Kuwait', ''): 'KWT',
             ('Kyrgyz Republic', ''): 'KGZ',
             ('Laos', ''): 'LAO',
             ('Latvia', ''): 'LVA',
             ('Lebanon', ''): 'LBN',
             ('Lesotho', ''): 'LSO',
             ('Liberia', ''): 'LBR',
             ('Libya', ''): 'LBY',
             ('Lithuania', ''): 'LTU',
             ('Luxembourg', ''): 'LUX',
             ('Macao', ''): 'MAC',
             ('Madagascar', ''): 'MDG',
             ('Malawi', ''): 'MWI',
             ('Malaysia', ''): 'MYS',
             ('Mali', ''): 'MLI',
             ('Mauritania', ''): 'MRT',
             ('Mauritius', ''): 'MUS',
             ('Mexico', ''): 'MEX',
             ('Moldova', ''): 'MDA',
             ('Monaco', ''): 'MCO',
             ('Mongolia', ''): 'MNG',
             ('Morocco', ''): 'MAR',
             ('Mozambique', ''): 'MOZ',
             ('Myanmar', ''): 'MMR',
             ('Namibia', ''): 'NAM',
             ('Nepal', ''): 'NPL',
             ('Netherlands', ''): 'NLD',
             ('New Zealand', ''): 'NZL',
             ('Nicaragua', ''): 'NIC',
             ('Niger', ''): 'NER',
             ('Nigeria', ''): 'NGA',
             ('Norway', ''): 'NOR',
             ('Oman', ''): 'OMN',
             ('Pakistan', ''): 'PAK',
             ('Palestine', ''): 'PSE',
             ('Panama', ''): 'PAN',
             ('Papua New Guinea', ''): 'PNG',
             ('Paraguay', ''): 'PRY',
             ('Peru', ''): 'PER',
             ('Philippines', ''): 'PHL',
             ('Poland', ''): 'POL',
             ('Portugal', ''): 'PRT',
             ('Puerto Rico', ''): 'PRI',
             ('Qatar', ''): 'QAT',
             ('Romania', ''): 'ROU',
             ('Russia', ''): 'RUS',
             ('Rwanda', ''): 'RWA',
             ('San Marino', ''): 'SMR',
             ('Saudi Arabia', ''): 'SAU',
             ('Senegal', ''): 'SEN',
             ('Serbia', ''): 'SRB',
             ('Seychelles', ''): 'SYC',
             ('Sierra Leone', ''): 'SLE',
             ('Singapore', ''): 'SGP',
             ('Slovak Republic', ''): 'SVK',
             ('Slovenia', ''): 'SVN',
             ('Solomon Islands', ''): 'SLB',
             ('Somalia', ''): 'SOM',
             ('South Africa', ''): 'ZAF',
             ('South Korea', ''): 'KOR',
             ('South Sudan', ''): 'SSD',
             ('Spain', ''): 'ESP',
             ('Sri Lanka', ''): 'LKA',
             ('Sudan', ''): 'SDN',
             ('Suriname', ''): 'SUR',
             ('Sweden', ''): 'SWE',
             ('Switzerland', ''): 'CHE',
             ('Syria', ''): 'SYR',
             ('Taiwan', ''): 'TWN',
             ('Tajikistan', ''): 'TJK',
             ('Tanzania', ''): 'TZA',
             ('Thailand', ''): 'THA',
             ('Timor-Leste', ''): 'TLS',
             ('Togo', ''): 'TGO',
             ('Trinidad and Tobago', ''): 'TTO',
             ('Tunisia', ''): 'TUN',
             ('Turkey', ''): 'TUR',
             ('Uganda', ''): 'UGA',
             ('Ukraine', ''): 'UKR',
             ('United Arab Emirates', ''): 'ARE',
             ('United Kingdom', ''): 'GBR',
             ('United Kingdom', 'England'): 'UK_ENG',
             ('United Kingdom', 'Northern Ireland'): 'UK_NIR',
             ('United Kingdom', 'Scotland'): 'UK_SCO',
             ('United Kingdom', 'Wales'): 'UK_WAL',
             ('United States', ''): 'USA',
             ('United States', 'Alabama'): 'US_AL',
             ('United States', 'Alaska'): 'US_AK',
             ('United States', 'Arizona'): 'US_AZ',
             ('United States', 'Arkansas'): 'US_AR',
             ('United States', 'California'): 'US_CA',
             ('United States', 'Colorado'): 'US_CO',
             ('United States', 'Connecticut'): 'US_CT',
             ('United States', 'Delaware'): 'US_DE',
             ('United States', 'Florida'): 'US_FL',
             ('United States', 'Georgia'): 'US_GA',
             ('United States', 'Hawaii'): 'US_HI',
             ('United States', 'Idaho'): 'US_ID',
             ('United States', 'Illinois'): 'US_IL',
             ('United States', 'Indiana'): 'US_IN',
             ('United States', 'Iowa'): 'US_IA',
             ('United States', 'Kansas'): 'US_KS',
             ('United States', 'Kentucky'): 'US_KY',
             ('United States', 'Louisiana'): 'US_LA',
             ('United States', 'Maine'): 'US_ME',
             ('United States', 'Maryland'): 'US_MD',
             ('United States', 'Massachusetts'): 'US_MA',
             ('United States', 'Michigan'): 'US_MI',
             ('United States', 'Minnesota'): 'US_MN',
             ('United States', 'Mississippi'): 'US_MS',
             ('United States', 'Missouri'): 'US_MO',
             ('United States', 'Montana'): 'US_MT',
             ('United States', 'Nebraska'): 'US_NE',
             ('United States', 'Nevada'): 'US_NV',
             ('United States', 'New Hampshire'): 'US_NH',
             ('United States', 'New Jersey'): 'US_NJ',
             ('United States', 'New Mexico'): 'US_NM',
             ('United States', 'New York'): 'US_NY',
             ('United States', 'North Carolina'): 'US_NC',
             ('United States', 'North Dakota'): 'US_ND',
             ('United States', 'Ohio'): 'US_OH',
             ('United States', 'Oklahoma'): 'US_OK',
             ('United States', 'Oregon'): 'US_OR',
             ('United States', 'Pennsylvania'): 'US_PA',
             ('United States', 'Rhode Island'): 'US_RI',
             ('United States', 'South Carolina'): 'US_SC',
             ('United States', 'South Dakota'): 'US_SD',
             ('United States', 'Tennessee'): 'US_TN',
             ('United States', 'Texas'): 'US_TX',
             ('United States', 'Utah'): 'US_UT',
             ('United States', 'Vermont'): 'US_VT',
             ('United States', 'Virgin Islands'): 'US_VI',
             ('United States', 'Virginia'): 'US_VA',
             ('United States', 'Washington'): 'US_WA',
             ('United States', 'Washington DC'): 'US_DC',
             ('United States', 'West Virginia'): 'US_WV',
             ('United States', 'Wisconsin'): 'US_WI',
             ('United States', 'Wyoming'): 'US_WY',
             ('Uruguay', ''): 'URY',
             ('Uzbekistan', ''): 'UZB',
             ('Vanuatu', ''): 'VUT',
             ('Venezuela', ''): 'VEN',
             ('Vietnam', ''): 'VNM',
             ('Yemen', ''): 'YEM',
             ('Zambia', ''): 'ZMB',
             ('Zimbabwe', ''): 'ZWE'}

VALID_AREAS = {v: k for k, v in REG_C_MAP.items()}
VALID_COUNTRIES = {k:v[0] for k,v in VALID_AREAS.items() if "_" not in k}
VALID_REGIONS =  {k:v for k,v in VALID_AREAS.items() if "_" in k}
COUNTRY_CODES = set(VALID_COUNTRIES.keys())
REGION_CODES = set(VALID_REGIONS.keys())

INV_VALID_COUNTRIES = {v: k for k, v in VALID_AREAS.items() if "_" not in k}
INV_VALID_REGIONS = {v: k for k, v in VALID_AREAS.items() if "_" in k}


from pprint import pprint
#pprint(list(VALID_COUNTRIES.values()))
#pprint(list(VALID_REGIONS.values()))
def filter_df_regions(oxford_df):

    ## code for filtering out rows that dont belong to set of countries and region we care

    #countries = list(VALID_COUNTRIES.values())
    #regions = VALID_REGIONS
    #oxford_df = oxford_df[(oxford_df.CountryName.isin(countries)) &
    #                      (oxford_df.RegionName.isin(regions))]

    areas = list(VALID_AREAS.values())

    oxford_df["RegionName"] = oxford_df["RegionName"].fillna(value="")
    mask = oxford_df[["CountryName", "RegionName"]].agg(tuple, 1).isin(areas)
    ret = oxford_df[mask]
    #pprint(ret.RegionName.unique())
    #pprint(ret.CountryName.unique())
    #pprint(ret.RegionName.nunique())
    #pprint(ret.CountryName.nunique())
    import numpy as np
    oxford_df["RegionName"] = oxford_df["RegionName"].replace(r'^\s*$', np.nan, regex=True)
    return ret


filter_df_regions(get_aug_oxford_df())