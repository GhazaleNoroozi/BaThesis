file_name = 'data.csv'

# 27 EU countries
EU_GEO_IDS = ['AT',
              'BE',
              'BG',
              'CY',
              'CZ',
              'DE',
              'DK',
              'EE',
              'EL',
              'ES',
              'FI',
              'FR',
              'HR',
              'HU',
              'IE',
              'IT',
              'LT',
              'LU',
              'LV',
              'MT',
              'NL',
              'PL',
              'PT',
              'RO',
              'SE',
              'SI',
              'SK']

# case_or_death = 'd'
CASE_OR_DEATH = 'c'

# what models to run
MODEL = {
    'arima': 1,
    'poly reg': 0,
    'rf reg': 0,
    'svr': 0
}

START_DATE = 1
DURATION = 14
END_DATE = START_DATE + DURATION
LAST_DATE = '14/10/2021'

ARIMA_ORDER = (7, 1, 7)
PR_DIMENSION = 2
SVR_GAMMA = 500
SVR_C = 10

ROTATION = 45
