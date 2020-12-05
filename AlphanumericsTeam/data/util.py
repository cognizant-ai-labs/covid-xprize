import os
from pprint import pprint
import pandas as pd
from pandas.io.parsers import read_csv
import csv

ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              os.pardir)
DATA_FILE_PATH = os.path.join(ROOT_DIRECTORY, "data", "files")


def get_country_codes():
    country_code_file  = os.path.join(DATA_FILE_PATH, "country_codes.txt")
    country_codes = set()
    with open(country_code_file) as f:
        for line in f:
            l = line.rstrip().split(",")
            country_codes.add((l[0], l[1] ))

    return country_codes


def get_region_codes():
    region_code_file  = os.path.join(DATA_FILE_PATH, "region_codes.txt")
    region_codes = set()
    with open(region_code_file) as f:
        for line in f:
            l = line.rstrip().split(",")
            region_codes.add((l[0], l[1], l[2], l[3] ))

    return region_codes


def get_orig_oxford_df():
    DATA_URL = os.path.join(DATA_FILE_PATH, "OxCGRT_latest.csv")
    oxford_df = pd.read_csv(DATA_URL,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                    "RegionCode": str},
                            error_bad_lines=False)

    oxford_df["RegionName"] = oxford_df["RegionName"].fillna("")
    return oxford_df


def get_aug_oxford_df():
    DATA_URL = os.path.join(DATA_FILE_PATH, "OxCGRT_latest_aug.csv")
    oxford_df = pd.read_csv(DATA_URL,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                    "RegionCode": str},
                            error_bad_lines=False)

    oxford_df["RegionName"] = oxford_df["RegionName"].fillna("")
    return oxford_df

def get_pop_df():
    DATA_URL = os.path.join(DATA_FILE_PATH, "pop.csv")
    pop_df = pd.read_csv(DATA_URL)
    return pop_df

df = get_pop_df()
#print(df[df["Code"]== "USA"].values.tolist()[0][2:5])

def get_valid_areas():

    DATA_URL = os.path.join(DATA_FILE_PATH, "countries_regions.csv")

    country_dict = {}
    region_dict = {}

    with open(DATA_URL) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            if row["CountryCode"]:
                country_dict[row["CountryCode"]] = row['CountryName']
            if row["RegionCode"]:
                region_dict[row["RegionCode"]] = (row['CountryName'], row['RegionName'])

    return country_dict, region_dict

VALID_COUNTRIES =  {'ABW': 'Aruba',
 'AFG': 'Afghanistan',
 'AGO': 'Angola',
 'ALB': 'Albania',
 'AND': 'Andorra',
 'ARE': 'United Arab Emirates',
 'ARG': 'Argentina',
 'AUS': 'Australia',
 'AUT': 'Austria',
 'AZE': 'Azerbaijan',
 'BDI': 'Burundi',
 'BEL': 'Belgium',
 'BEN': 'Benin',
 'BFA': 'Burkina Faso',
 'BGD': 'Bangladesh',
 'BGR': 'Bulgaria',
 'BHR': 'Bahrain',
 'BHS': 'Bahamas',
 'BIH': 'Bosnia and Herzegovina',
 'BLR': 'Belarus',
 'BLZ': 'Belize',
 'BMU': 'Bermuda',
 'BOL': 'Bolivia',
 'BRA': 'Brazil',
 'BRB': 'Barbados',
 'BRN': 'Brunei',
 'BTN': 'Bhutan',
 'BWA': 'Botswana',
 'CAF': 'Central African Republic',
 'CAN': 'Canada',
 'CHE': 'Switzerland',
 'CHL': 'Chile',
 'CHN': 'China',
 'CIV': "Cote d'Ivoire",
 'CMR': 'Cameroon',
 'COD': 'Democratic Republic of Congo',
 'COG': 'Congo',
 'COL': 'Colombia',
 'COM': 'Comoros',
 'CPV': 'Cape Verde',
 'CRI': 'Costa Rica',
 'CUB': 'Cuba',
 'CYP': 'Cyprus',
 'CZE': 'Czech Republic',
 'DEU': 'Germany',
 'DJI': 'Djibouti',
 'DMA': 'Dominica',
 'DNK': 'Denmark',
 'DOM': 'Dominican Republic',
 'DZA': 'Algeria',
 'ECU': 'Ecuador',
 'EGY': 'Egypt',
 'ERI': 'Eritrea',
 'ESP': 'Spain',
 'EST': 'Estonia',
 'ETH': 'Ethiopia',
 'FIN': 'Finland',
 'FJI': 'Fiji',
 'FRA': 'France',
 'FRO': 'Faeroe Islands',
 'GAB': 'Gabon',
 'GBR': 'United Kingdom',
 'GEO': 'Georgia',
 'GHA': 'Ghana',
 'GIN': 'Guinea',
 'GMB': 'Gambia',
 'GRC': 'Greece',
 'GRL': 'Greenland',
 'GTM': 'Guatemala',
 'GUM': 'Guam',
 'GUY': 'Guyana',
 'HKG': 'Hong Kong',
 'HND': 'Honduras',
 'HRV': 'Croatia',
 'HTI': 'Haiti',
 'HUN': 'Hungary',
 'IDN': 'Indonesia',
 'IND': 'India',
 'IRL': 'Ireland',
 'IRN': 'Iran',
 'IRQ': 'Iraq',
 'ISL': 'Iceland',
 'ISR': 'Israel',
 'ITA': 'Italy',
 'JAM': 'Jamaica',
 'JOR': 'Jordan',
 'JPN': 'Japan',
 'KAZ': 'Kazakhstan',
 'KEN': 'Kenya',
 'KGZ': 'Kyrgyz Republic',
 'KHM': 'Cambodia',
 'KOR': 'South Korea',
 'KWT': 'Kuwait',
 'LAO': 'Laos',
 'LBN': 'Lebanon',
 'LBR': 'Liberia',
 'LBY': 'Libya',
 'LKA': 'Sri Lanka',
 'LSO': 'Lesotho',
 'LTU': 'Lithuania',
 'LUX': 'Luxembourg',
 'LVA': 'Latvia',
 'MAC': 'Macao',
 'MAR': 'Morocco',
 'MCO': 'Monaco',
 'MDA': 'Moldova',
 'MDG': 'Madagascar',
 'MEX': 'Mexico',
 'MLI': 'Mali',
 'MMR': 'Myanmar',
 'MNG': 'Mongolia',
 'MOZ': 'Mozambique',
 'MRT': 'Mauritania',
 'MUS': 'Mauritius',
 'MWI': 'Malawi',
 'MYS': 'Malaysia',
 'NAM': 'Namibia',
 'NER': 'Niger',
 'NGA': 'Nigeria',
 'NIC': 'Nicaragua',
 'NLD': 'Netherlands',
 'NOR': 'Norway',
 'NPL': 'Nepal',
 'NZL': 'New Zealand',
 'OMN': 'Oman',
 'PAK': 'Pakistan',
 'PAN': 'Panama',
 'PER': 'Peru',
 'PHL': 'Philippines',
 'PNG': 'Papua New Guinea',
 'POL': 'Poland',
 'PRI': 'Puerto Rico',
 'PRT': 'Portugal',
 'PRY': 'Paraguay',
 'PSE': 'Palestine',
 'QAT': 'Qatar',
 'RKS': 'Kosovo',
 'ROU': 'Romania',
 'RUS': 'Russia',
 'RWA': 'Rwanda',
 'SAU': 'Saudi Arabia',
 'SDN': 'Sudan',
 'SEN': 'Senegal',
 'SGP': 'Singapore',
 'SLB': 'Solomon Islands',
 'SLE': 'Sierra Leone',
 'SLV': 'El Salvador',
 'SMR': 'San Marino',
 'SOM': 'Somalia',
 'SRB': 'Serbia',
 'SSD': 'South Sudan',
 'SUR': 'Suriname',
 'SVK': 'Slovak Republic',
 'SVN': 'Slovenia',
 'SWE': 'Sweden',
 'SWZ': 'Eswatini',
 'SYC': 'Seychelles',
 'SYR': 'Syria',
 'TCD': 'Chad',
 'TGO': 'Togo',
 'THA': 'Thailand',
 'TJK': 'Tajikistan',
 'TLS': 'Timor-Leste',
 'TTO': 'Trinidad and Tobago',
 'TUN': 'Tunisia',
 'TUR': 'Turkey',
 'TWN': 'Taiwan',
 'TZA': 'Tanzania',
 'UGA': 'Uganda',
 'UKR': 'Ukraine',
 'URY': 'Uruguay',
 'USA': 'United States',
 'UZB': 'Uzbekistan',
 'VEN': 'Venezuela',
 'VNM': 'Vietnam',
 'VUT': 'Vanuatu',
 'YEM': 'Yemen',
 'ZAF': 'South Africa',
 'ZMB': 'Zambia',
 'ZWE': 'Zimbabwe'}


VALID_REGIONS = {'UK_ENG': ('United Kingdom', 'England'),
 'UK_NIR': ('United Kingdom', 'Northern Ireland'),
 'UK_SCO': ('United Kingdom', 'Scotland'),
 'UK_WAL': ('United Kingdom', 'Wales'),
 'US_AK': ('United States', 'Alaska'),
 'US_AL': ('United States', 'Alabama'),
 'US_AR': ('United States', 'Arkansas'),
 'US_AZ': ('United States', 'Arizona'),
 'US_CA': ('United States', 'California'),
 'US_CO': ('United States', 'Colorado'),
 'US_CT': ('United States', 'Connecticut'),
 'US_DC': ('United States', 'Washington DC'),
 'US_DE': ('United States', 'Delaware'),
 'US_FL': ('United States', 'Florida'),
 'US_GA': ('United States', 'Georgia'),
 'US_HI': ('United States', 'Hawaii'),
 'US_IA': ('United States', 'Iowa'),
 'US_ID': ('United States', 'Idaho'),
 'US_IL': ('United States', 'Illinois'),
 'US_IN': ('United States', 'Indiana'),
 'US_KS': ('United States', 'Kansas'),
 'US_KY': ('United States', 'Kentucky'),
 'US_LA': ('United States', 'Louisiana'),
 'US_MA': ('United States', 'Massachusetts'),
 'US_MD': ('United States', 'Maryland'),
 'US_ME': ('United States', 'Maine'),
 'US_MI': ('United States', 'Michigan'),
 'US_MN': ('United States', 'Minnesota'),
 'US_MO': ('United States', 'Missouri'),
 'US_MS': ('United States', 'Mississippi'),
 'US_MT': ('United States', 'Montana'),
 'US_NC': ('United States', 'North Carolina'),
 'US_ND': ('United States', 'North Dakota'),
 'US_NE': ('United States', 'Nebraska'),
 'US_NH': ('United States', 'New Hampshire'),
 'US_NJ': ('United States', 'New Jersey'),
 'US_NM': ('United States', 'New Mexico'),
 'US_NV': ('United States', 'Nevada'),
 'US_NY': ('United States', 'New York'),
 'US_OH': ('United States', 'Ohio'),
 'US_OK': ('United States', 'Oklahoma'),
 'US_OR': ('United States', 'Oregon'),
 'US_PA': ('United States', 'Pennsylvania'),
 'US_RI': ('United States', 'Rhode Island'),
 'US_SC': ('United States', 'South Carolina'),
 'US_SD': ('United States', 'South Dakota'),
 'US_TN': ('United States', 'Tennessee'),
 'US_TX': ('United States', 'Texas'),
 'US_UT': ('United States', 'Utah'),
 'US_VA': ('United States', 'Virginia'),
 'US_VI': ('United States', 'Virgin Islands'),
 'US_VT': ('United States', 'Vermont'),
 'US_WA': ('United States', 'Washington'),
 'US_WI': ('United States', 'Wisconsin'),
 'US_WV': ('United States', 'West Virginia'),
 'US_WY': ('United States', 'Wyoming')}

COUNTRY_CODES = set(VALID_COUNTRIES.keys())
REGION_CODES = set(VALID_REGIONS.keys())