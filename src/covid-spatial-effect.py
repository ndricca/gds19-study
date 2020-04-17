import geopandas as gpd
import pandas as pd

from src.config import GH_PRV_SHP_URL, GH_COVID_URL_TMP


def get_prov_shp():
    prov_shp = gpd.read_file(GH_PRV_SHP_URL)
    return prov_shp


def get_prov_df(date=None):
    if date is None:
        date = pd.Timestamp.now().strftime('%Y%m%d')  # 20200417
    prov_df = pd.read_csv(GH_COVID_URL_TMP.format(date))


if __name__ == '__main__':
    prov_shp = get_prov_shp()
