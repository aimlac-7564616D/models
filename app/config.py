from os import environ, urandom

SECRET_KEY = urandom(1024)

DATETIME_FORMAT = "%Y-%m-%dT%H:%MZ"
LATITUDE = environ.get('LOCATION_LAT')
LONGITUDE = environ.get('LOCATION_LON')
TIMEZONE = 'Europe/London'

ALTITUDE = 250.0            # m above sea level
PANEL_TILT = 45.0           # angle panels are tilted at (south facing)
ARRAY_AREA = 50.0 * 50.0    # area covered by array in m^2
BASE_EFFICIENCY = 0.196     # base efficiency of panels
PMPP = -0.0037              # %/C
PMAX_ARRAY = 469_000        # W