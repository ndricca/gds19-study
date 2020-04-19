import contextily as ctx
from matplotlib import pyplot as plt
from mapclassify import Quantiles, EqualInterval, FisherJenks, NaturalBreaks
import numpy as np
import seaborn as sns
import imageio

from src.config import FIGS_DIR


def plot_scheme(scheme, var, db, k=7, figsize=(16, 8), saveto=None):
    """
    Plot the distribution over value and geographical space of variable `var` using scheme `scheme`
    :param scheme: name of the classification scheme to use
    :param var: variable name
    :param db: table with input data
    :param k: number of bins
    :param figsize: size of the figure to be created
    :param saveto: path for file to save the plot
    """
    schemes = {'equalinterval': EqualInterval,
               'quantiles': Quantiles,
               'fisherjenks': FisherJenks,
               'naturalbreaks': NaturalBreaks}
    try:
        sch = schemes[scheme]
    except KeyError as e:
        raise RuntimeError("invalid scheme {e}, use one of {l}".format(e=e, l=schemes.keys()))
    classi = schemes[scheme](db[var], k=k)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # KDE
    sns.kdeplot(db[var], shade=True, color='purple', ax=ax1)
    sns.rugplot(db[var], alpha=0.5, color='purple', ax=ax1)
    for cut in classi.bins:
        ax1.axvline(cut, color='blue', linewidth=0.75)
    ax1.set_title('Value distribution')
    # Map
    p = db.plot(column=var, scheme=scheme, alpha=0.95, k=k, legend=True,
                cmap=plt.cm.RdPu, ax=ax2, linewidth=0.1, )
    ax2.axis('equal')
    ax2.set_axis_off()
    ax2.set_title('Geographical distribution')
    ctx.add_basemap(ax2, url=ctx.providers.Stamen.TonerLite)
    f.suptitle(scheme, size=25)
    if saveto:
        plt.savefig(saveto)
    plt.show()


def _plot_for_gif(df, date, scheme, var='totale_casi', k=4, x_max=20000, figsize=(16, 8)):
    db = df.loc[df['data'] == date]
    schemes = {
        'equalinterval': EqualInterval,
        'quantiles': Quantiles,
        'fisherjenks': FisherJenks,
        'naturalbreaks': NaturalBreaks,
        'globalquantiles': 'userdefined'
    }
    try:
        _ = schemes[scheme]
    except KeyError as e:
        raise RuntimeError("invalid scheme {e}, use one of {l}".format(e=e, l=schemes.keys()))
    if scheme == 'globalquantiles':
        classi = Quantiles(df[var], k=k)
        classification_kwds = {'bins': classi.bins.tolist()}
    else:
        classi = schemes[scheme](db[var], k=k)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # KDE
    sns.kdeplot(db[var], shade=True, color='purple', bw=5 / (np.max(db[var]) - np.min(db[var])), ax=ax1)
    sns.rugplot(db[var], alpha=0.5, color='purple', ax=ax1)
    for cut in classi.bins:
        ax1.axvline(cut, color='blue', linewidth=0.75)
    ax1.set_title('Value distribution')
    ax2.set_xlim(0, x_max)
    if scheme == 'globalquantiles':
        # Map
        p = db.plot(column=var, scheme=schemes[scheme], classification_kwds=classification_kwds, alpha=0.95, k=k,
                    legend=True, cmap=plt.cm.RdPu, ax=ax2, linewidth=0.1)
    else:
        p = db.plot(column=var, scheme=scheme, alpha=0.95, k=k, legend=True, cmap=plt.cm.RdPu, ax=ax2, linewidth=0.1)
    ax2.axis('equal')
    ax2.set_axis_off()
    ax2.set_title('Geographical distribution')
    # ctx.add_basemap(ax2, url=ctx.providers.Stamen.TonerLite)
    f.suptitle("As of " + np.datetime_as_string(date, unit='D'), size=25)
    f.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
    return image


def create_gif(df, scheme, var_col, k, dt_col='data'):
    """
    Save a gif file with day by day biplot of data using the specified scheme and k bins.
    If scheme is set as globalquantiles, a unique scheme is used among all dates
    :param df: geopandas dataframe with all observations
    :param scheme: string that specifies which scheme will be used to create bins
    :param var_col: column with values
    :param k: number of bins
    :param dt_col: column with dates
    """
    dates = df[dt_col].dropna().unique().tolist()
    gif_file = "_".join([scheme] + [d.strftime('%Y%m%d') for d in [df['data'].max(), df['data'].min()]]) + ".gif"
    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    imageio.mimsave(
        os.path.join(FIGS_DIR, gif_file),
        [_plot_for_gif(df=df, date=d, var=var_col, scheme=scheme, x_max=4000, k=k) for d in dates],
        fps=1)


if __name__ == '__main__':
    import logging
    import os
    from src.load_covid_data_funcs import load_data_with_shp

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logging.info(os.getcwd())
    date = '20200311'

    path_fig = os.path.join(FIGS_DIR, 'plot_scheme_quantiles_{}.png'.format(date))
    prv = load_data_with_shp(date=date)
    plt = plot_scheme(scheme='quantiles', var='totale_casi', db=prv, k=5, saveto=path_fig)
