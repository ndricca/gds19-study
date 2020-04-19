import os

import geopandas as gpd
import pandas as pd
import logging

from src.config import GH_PRV_SHP_URL, GH_COVID_SINGLE_URL_TMP, GH_COVID_ALL_URL, COVID_DATA_DIR


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
    logging.info("loading covid data from " + GH_COVID_SINGLE_URL_TMP.format(date))
    prov_df = pd.read_csv(GH_COVID_SINGLE_URL_TMP.format(date))
    return prov_df


def get_prov_all_dates():
    """
    Download covid csv data for the all dates available and return a dataframe.
    :return: pandas dataframe with covid data for selected all dates
    """
    logging.info("loading covid data from " + GH_COVID_ALL_URL)
    prov_all_dt = pd.read_csv(GH_COVID_ALL_URL)
    prov_all_dt['data'] = pd.to_datetime(prov_all_dt.loc[:, 'data'], errors='ignore').dt.floor('D').values
    return prov_all_dt


def read_tot_pop_prov():
    """
    Read ISTAT csv total population data by provinces and return a dataframe.
    :return: pandas dataframe with total population data by provinces
    """
    tot_pop_path = os.path.join(COVID_DATA_DIR, 'ISTAT_PROV_POP_TOT.csv')
    logging.info("loading istat data from " + tot_pop_path)
    tot_pop_df = pd.read_csv(tot_pop_path)
    return tot_pop_df


def filter_tot_pop(tot_pop_df):
    """
    Create a dataframe with total population by province and percentage of population by 5 age groups
    :param tot_pop_df: ISTAT total dataframe
    :return: pandas dataframe
    """
    filt_tot = tot_pop_df.copy()
    prov_tot = filt_tot.loc[filt_tot['ITTER107'].str.len() == 5]
    sex_tot = prov_tot.loc[prov_tot.SEXISTAT1 == 9]
    st_civ_tot = sex_tot.loc[sex_tot.STATCIV2 == 99].reset_index(drop=True)
    st_civ_tot['eta2'] = st_civ_tot.ETA1.str.extract('(\d+)').astype(float).floordiv(20)
    prov_tot = st_civ_tot.loc[~st_civ_tot['eta2'].isna()].reset_index(drop=True)
    prov_tot.loc[prov_tot['eta2'] == 5, 'eta2'] = 4
    bin_labels = ['Y0-Y19', 'Y20-Y39', 'Y40-Y59', 'Y60-Y79', 'Y80+']
    prov_tot['ETA_BINS'] = prov_tot['eta2'].replace({i: l for i, l in enumerate(bin_labels)}).values
    pivot_prov = prov_tot.pivot_table(index='Territorio', columns='ETA_BINS', values='Value')
    pivot_prov['TOT'] = pivot_prov.sum(axis=1)
    pivot_prov.loc[:, bin_labels] = pivot_prov.loc[:, bin_labels].div(pivot_prov['TOT'], axis=0)
    return pivot_prov.reset_index()


def merge_istat_covid_df(prov_df, istat_df):
    """
    Merge on province name, with some manual correction
    :param prov_df: covid dataframe
    :param prov_tot_df: istat dataframe
    :return: merged dataframe
    """
    rename_dict = {
        "Valle d'Aosta / Vallée d'Aoste": "Aosta",
        "Bolzano / Bozen": "Bolzano",
        "Massa-Carrara": "Massa Carrara"
    }
    istat_df['Territorio'] = istat_df['Territorio'].replace(rename_dict)
    istat_df = istat_df.rename(columns={'Territorio': 'denominazione_provincia'})
    merged_df = pd.merge(prov_df, istat_df, on='denominazione_provincia')
    return merged_df


def calculate_pop_features(merged_df):
    merged_with_feat_df = merged_df.copy()
    elder_labels = ['Y60-Y79', 'Y80+']
    merged_with_feat_df['PERC_ANZIANI'] = merged_with_feat_df.loc[:, elder_labels].sum(axis=1)
    merged_with_feat_df['CASI_POP_TOT'] = merged_with_feat_df['totale_casi'] / merged_with_feat_df['TOT']
    merged_with_feat_df['CASI_POP_60+'] = merged_with_feat_df['totale_casi'] / merged_with_feat_df['TOT'] * merged_with_feat_df['PERC_ANZIANI']
    return merged_with_feat_df


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
    """
    Pipeline to produce the dataframe that will be visualized for selected date
    :param date: string with format YYYYmmdd (pythonic format: "%Y%m%d")
    :return: covid geopandas dataframe with shape for selected date
    """
    if date is None:
        date = pd.Timestamp.now().strftime('%Y%m%d')  # 20200417
    prov_shp = get_prov_shp()
    prov_df = get_prov_df(date=date)
    df_with_shp = merge_df_with_shp(prov_df=prov_df, prov_shp=prov_shp)
    return df_with_shp


def load_all_data_with_shp():
    """
    Pipeline to produce the dataframe that will be visualized for all dates available
    :return: covid geopandas dataframe with shape for all dates available
    """
    istat_raw = read_tot_pop_prov()
    istat_df = filter_tot_pop(tot_pop_df=istat_raw)
    prov_df = get_prov_all_dates()
    merged_prov_df = merge_istat_covid_df(prov_df=prov_df,istat_df=istat_df)
    enriched_prov_df = calculate_pop_features(merged_df=merged_prov_df)
    prov_shp = get_prov_shp()
    df_with_shp = merge_df_with_shp(prov_df=enriched_prov_df, prov_shp=prov_shp)
    return df_with_shp


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # date = '20200417'
    # prov_shp = get_prov_shp()
    # prov_df = get_prov_df(date=date)
    # df_with_shp = merge_df_with_shp(prov_df=prov_df, prov_shp=prov_shp)
    df_with_shp = load_all_data_with_shp()
    logging.info("loading successful, with shape {}".format(df_with_shp.shape))
