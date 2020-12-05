from collections import defaultdict
import holidays
from dateutil.parser import parse


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