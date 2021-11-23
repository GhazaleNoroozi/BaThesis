file_name = 'data.csv'

# 27 EU countries
EU_GEO_IDS = [
              'AT',
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
              'SK'
            ]
# case_or_death = 'd'
CASE_OR_DEATH = 'c'

# what models to run
MODEL = {
    'arima': 1,
    'poly reg': 1,
    'rf reg': 1,
    'svr': 1
}

START_DATE = 1
DURATION = 14
END_DATE = START_DATE + DURATION
LAST_DATE = '14/10/2021'

# ARIMA_ORDER = (7, 1, 7)
ARIMA_ORDER = (6, 2, 6)

SARIMA_TREND = (2, 1, 0)
SARIMA_SEASONAL = (1, 1, 0, 7)

PR_DIMENSION = 5

SVR_GAMMA = 'scale'
SVR_C = 10

ROTATION = 50
