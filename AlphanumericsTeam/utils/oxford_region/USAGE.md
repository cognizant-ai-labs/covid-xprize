 **region_stats.py** script can be used to extract country or region specific information from the oxford dataset.

 The script requires access to internet as the dataset is downloaded from the official [source!] (https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv)

**usage: region_stats.py [-h] (-r REGION_CODE | -c COUNTRY_CODE) -s START_DATE  -e END_DATE  [-ho]**

optional arguments:
  -h, --help            show this help message and exit

  -r REGION_CODE, --region-code REGION_CODE
                        region to select data from

  -c COUNTRY_CODE, --country-code COUNTRY_CODE
                        country to select data from

  -s START_DATE, --start-date START_DATE
                        the start date (in yyyy-mm-dd) to begin getting data
                        from

  -e END_DATE, --end-date END_DATE
                        the last date (in yyyy-mm-dd) to get data from (note:
                        last date is included)

  -ho, --holidays       request additional column for holidays/weekend

NOTE: One of region code or country code is expected as argument but not both.

OUTPUT: a csv file in the current directory with name "countrycode__regioncode_fromdate_todate.csv"