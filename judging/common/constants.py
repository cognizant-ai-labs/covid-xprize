class Constants:
    # Bucket to be used for prediction results, ranking info etc.
    S3_BUCKET = 'covid-xprize'

    # Source for NPI data
    LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'

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