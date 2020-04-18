from matplotlib import pyplot as plt
from mapclassify import Quantiles, EqualInterval, FisherJenks
import seaborn as sns


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
               'fisherjenks': FisherJenks}
    classi = schemes[scheme](db[var], k=7)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # KDE
    sns.kdeplot(db[var], shade=True, color='purple', ax=ax1)
    sns.rugplot(db[var], alpha=0.5, color='purple', ax=ax1)
    for cut in classi.bins:
        ax1.axvline(cut, color='blue', linewidth=0.75)
    ax1.set_title('Value distribution')
    # Map
    p = db.plot(column=var, scheme=scheme, alpha=0.75, k=7, legend=True,
                cmap=plt.cm.RdPu, ax=ax2, linewidth=0.1)
    ax2.axis('equal')
    ax2.set_axis_off()
    ax2.set_title('Geographical distribution')
    f.suptitle(scheme, size=25)
    if saveto:
        plt.savefig(saveto)
    return plt


if __name__ == '__main__':
    import logging
    from src.load_covid_data_funcs import load_data_with_shp

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    date = '20200402'
    path_fig = '..\\figs\\plot_scheme_quantiles_{}.png'.format(date)
    prv = load_data_with_shp(date=date)
    plt = plot_scheme(scheme='quantiles', var='totale_casi', db=prv, k=5, saveto=path_fig)
