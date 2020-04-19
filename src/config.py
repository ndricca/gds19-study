import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGS_DIR = os.path.join(PROJECT_DIR, 'figs')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
COVID_DATA_DIR = os.path.join(DATA_DIR, 'covid_italy')

GH_PRV_SHP_URL = 'https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_provinces.geojson'
GH_COVID_SINGLE_URL_TMP = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province-{}.csv'
GH_COVID_ALL_URL = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv'


