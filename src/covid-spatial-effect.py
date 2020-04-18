import geopandas as gpd
import pandas as pd
import logging

from src.config import GH_PRV_SHP_URL, GH_COVID_URL_TMP


def get_prov_shp():
    """
    Download geojson data at province level and return geopandas dataframe
    :return: geopandas dataframe with province level geometry
    """
    logging.info("loading shape from " + GH_PRV_SHP_URL)
    prov_shp = gpd.read_file(GH_PRV_SHP_URL)
    prov_shp = prov_shp.set_index('prov_acr')
    return prov_shp


def get_prov_df(date=None):
    """
    Download covid csv data for the selected date and return a dataframe.
    :param date: string with format YYYYmmdd (pythonic format: "%Y%m%d")
    :return: pandas dataframe with covid data for selected date
    """
    if date is None:
        date = pd.Timestamp.now().strftime('%Y%m%d')  # 20200417
    logging.info("loading covid data from " + GH_COVID_URL_TMP.format(date))
    prov_df = pd.read_csv(GH_COVID_URL_TMP.format(date))
    return prov_df


def merge_df_with_shp(prov_df, prov_shp):
    """
    Merge dataframes to assign shape to covid dataframe
    :param prov_df: covid pandas dataframe
    :param prov_shp: geopandas dataframe
    :return: covid geopandas dataframe
    """
    logging.info("merging dataframe using 'prov_acr' as index")
    prov_df = prov_df.rename(columns={'sigla_provincia': 'prov_acr'})
    prov_df = prov_df.set_index('prov_acr')
    df_merged = prov_shp.join(prov_df, on='prov_acr')
    return df_merged


def load_data_with_shp(date=None):
    if date is None:
        date = pd.Timestamp.now().strftime('%Y%m%d')  # 20200417
    prov_shp = get_prov_shp()
    prov_df = get_prov_df(date=date)
    df_with_shp = merge_df_with_shp(prov_df=prov_df, prov_shp=prov_shp)
    return df_with_shp


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    date = '20200417'
    prov_shp = get_prov_shp()
    prov_df = get_prov_df(date=date)
    df_with_shp = merge_df_with_shp(prov_df=prov_df, prov_shp=prov_shp)
    logging.info("loading successful, with shape {}".format(df_with_shp.shape))
