# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

"""
For metadata relating to NPIs -- descriptions, phases etc.
"""
from collections import OrderedDict

NPI_COLUMNS = [
    'C1_School closing',
    'C2_Workplace closing',
    'C3_Cancel public events',
    'C4_Restrictions on gatherings',
    'C5_Close public transport',
    'C6_Stay at home requirements',
    'C7_Restrictions on internal movement',
    'C8_International travel controls',
    'H1_Public information campaigns',
    'H2_Testing policy',
    'H3_Contact tracing',
    'H6_Facial Coverings'
]

ERROR_COLUMNS = ['UpperNewCases', 'LowerNewCases']

DISPLAY_COLUMNS = (['CountryName', 'Date', 'ConfirmedCases', 'NewCases',
                   'NoNpiNewCases', 'MaxNpisNewCases', 'CurrentNpisNewCases', 'Forecast', "HistoricalNewCases"]
                   + NPI_COLUMNS + ERROR_COLUMNS)

NPI_DETAILS = [
    [
        '0: None',
        '1: Mild',
        '2: Medium',
        '3: Strict'
    ],
    [
        '0: None',
        '1: Mild',
        '2: Medium',
        '3: Strict'
    ],
    [
        '0: None',
        '1: Medium',
        '2: Strict'
    ],
    [
        '0: None',
        '1: Mild',
        '2: Medium',
        '3: Hard',
        '4: Strict'
    ],
    [
        '0: None',
        '1: Medium',
        '2: Strict'
    ],
    [
        '0: None',
        '1: Mild',
        '2: Medium',
        '3: Strict'
    ],
    [
        '0: None',
        '1: Medium',
        '2: Strict'
    ],
    [
        '0: None',
        '1: Mild',
        '2: Medium',
        '3: Hard',
        '4: Strict'
    ],
    [
        '0: None',
        '1: Medium',
        '2: Strict'
    ],
    [
        '0: None',
        '1: Mild',
        '2: Medium',
        '3: Strict'
    ],
    [
        '0: None',
        '1: Medium',
        '2: Strict'
    ],
    [
        '0: None',
        '1: Mild',
        '2: Medium',
        '3: Hard',
        '4: Strict'
    ]
]

# Color palette for various NPIs in the Gantt chart.
# NPIs have different ranges (0-2, 0-3, 0-4) and this dictionary allows us to color the bars appropriately for each NPI.
NPI_COLORS = {
    '0: None'    : '#6DCBFF',
    '1: Mild'    : '#328DFF',
    '1: Medium'  : '#2D67FF',
    '2: Medium'  : '#2D67FF',
    '2: Strict'  : '#000063',
    '3: Strict'  : '#000063',
    '3: Hard'    : '#0033a0',
    '4: Strict'  : '#000063'
}

NPI_PHASES = {
    'C1': {
        '0': 'No measures',
        '1': 'Recommend closing',
        '2': 'Require closing (only some levels or categories, eg just high school, or just public schools)',
        '3': 'Require closing all levels'
    },

    'C2': {
        '0': 'No measures',
        '1': 'Recommend closing (or recommend work from home)',
        '2': 'Require closing (or work from home) for some sectors or categories of workers',
        '3': 'Require closing (or work from home) for all but essential workplaces eg grocery stores doctors'
    },

    'C3': {
        '0': 'No measures',
        '1': 'Recommend cancelling',
        '2': 'Require cancelling'
    },

    'C4': {
        '0': 'No restrictions',
        '1': 'Restrictions on very large gatherings the limit is above 1000 people',
        '2': 'Restrictions on gatherings between 101 1000 people',
        '3': 'Restrictions on gatherings between 11 100 people',
        '4': 'Restrictions on gatherings of 10 people or less'
    },

    'C5': {
        '0': 'No measures',
        '1': 'Recommend closing or significantly reduce volume/route/means of transport available',
        '2': 'Require closing (or prohibit most citizens from using it)'
    },

    'C6': {
        '0': 'No measures',
        '1': 'Recommend not leaving house',
        '2': 'Require not leaving house with exceptions for daily exercise, grocery shopping, and ''essential'' trips',
        '3': 'Require not leaving house with minimal exceptions (eg allowed to leave once a week, or only one person can leave at a time, etc)'
    },

    'C7': {
        '0': 'No measures',
        '1': 'Recommend not to travel between regions cities',
        '2': 'Internal movement restrictions in place'
    },

    'C8': {
        '0': 'No restrictions',
        '1': 'Screening arrivals',
        '2': 'Quarantine arrivals from some or all regions',
        '3': 'Ban arrivals from some regions',
        '4': 'Ban on all regions or total border closure'
    },

    'H1': {
        '0': 'No COVID-19 public information campaign',
        '1': 'Public officials urging caution about COVID-19',
        '2': 'Coordinated public information campaign (e.g. across traditional and social media)'
    },

    'H2': {
        '0': 'No testing policy',
        '1': 'Only those who both (a) have symptoms AND (b) meet specific criteria (e.g. key workers, admitted to hospital, came into contact with a known case, returned from overseas)',
        '2': 'Testing of anyone showing COVID-19 symptoms',
        '3': 'Open public testing (e.g. “drive through” testing available to asymptomatic people)'
    },

    'H3': {
        '0': 'No contact tracing',
        '1': 'Limited contact tracing - not done for all cases',
        '2': 'Comprehensive contact tracing - done for all identified cases'
    },

    'H6': {
        '0': 'No policy',
        '1': 'Recommended',
        '2': 'Required in some specified shared/public spaces outside the home with other people present, or some situations when social distancing not possible',
        '3': 'Required in all shared/public spaces outside the home with other people present or all situations when social distancing not possible',
        '4': 'Required outside the home at all times regardless of location or presence of other people'
    }
}

LEGEND_COLORS = OrderedDict({
    'None': NPI_COLORS['0: None'],
    'Medium': NPI_COLORS['1: Medium'],
    'Hard': NPI_COLORS['3: Hard'],
    'Strict': NPI_COLORS['4: Strict'],
})

# Max values
MAX_C1_SCHOOL_CLOSING = 3
MAX_C2_WORKPLACE_CLOSING = 3
MAX_C3_CANCEL_PUBLIC_EVENTS = 2
MAX_C4_RESTRICTIONS_ON_GATHERINGS = 4
MAX_C5_CLOSE_PUBLIC_TRANSPORT = 2
MAX_C6_STAY_AT_HOME_REQUIREMENTS = 3
MAX_C7_RESTRICTIONS_ON_INTERNAL_MOVEMENT = 2
MAX_C8_INTERNATIONAL_TRAVEL_CONTROLS = 4
MAX_H1_PUBLIC_INFO_CAMPAIGNS = 2
MAX_H2_TESTING_POLICY = 3
MAX_H3_CONTACT_TRACING = 2
MAX_H6_FACIAL_COVERINGS = 4

# Max values as list
MAX_NPIS_VECTOR = [MAX_C1_SCHOOL_CLOSING, MAX_C2_WORKPLACE_CLOSING,
                   MAX_C3_CANCEL_PUBLIC_EVENTS, MAX_C4_RESTRICTIONS_ON_GATHERINGS,
                   MAX_C5_CLOSE_PUBLIC_TRANSPORT, MAX_C6_STAY_AT_HOME_REQUIREMENTS,
                   MAX_C7_RESTRICTIONS_ON_INTERNAL_MOVEMENT, MAX_C8_INTERNATIONAL_TRAVEL_CONTROLS,
                   MAX_H1_PUBLIC_INFO_CAMPAIGNS, MAX_H2_TESTING_POLICY, MAX_H3_CONTACT_TRACING,
                   MAX_H6_FACIAL_COVERINGS]