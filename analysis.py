import re
import os
import gc
import multiprocessing as mp
import shutil
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
import matplotlib as mpl
mpl.use('cairo') # must be called before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.font_manager as fm
from matplotlib.ticker import FixedLocator, MultipleLocator
from bokeh.palettes import Spectral9 # just a list of colors, can remove safely if you don't have bokeh

# Matplotlib will release a bombardment of harmless warnings without
# the following line. Proceed with caution, however, as this silences
# all warnings.
warnings.filterwarnings("ignore")

FRED_dir = ''
county_file = ''
dates_master = pd.date_range(start='1970-01-01', end='2013-01-01', freq='AS')


class DateError(Exception):
    pass


def get_county_dict(filename):
    """
    Parses text file with list of counties (or equivalents, e.g. parish,
    independent city) and FIPS codes into a dictionary with format
    {FIPS_code (string): County Name, ST} where ST is the standard 2 
    letter state code. File should be downloaded from US Census data 
    (https://www.census.gov/geo/reference/codes/cou.html), or otherwise
    in the exact same format, for example:

    AL,01,001,Autauga County,H1

    By default, the return dictionary will include all 50 states and DC,
    but not other territories.

    Parameters
    ----------
    filename : string
        String indicating filename of text file which has the proper
        format (see above).

    Returns
    -------
    counties : dictionary
        Dictionary with format {FIPS_code (string): County Name, ST}

    Note: There will be 3235 entries in the 2010 United States data set.

    """
    exclude = [
        'AS',
        'GU',
        'MP',
        'PR',
        'UM',
        'VI'
    ]
    with open(filename) as county_file:
        counties = [county.strip().split(",") for county in county_file.readlines()]
        counties = {"".join(county[1:3]) : county[3]+", "+county[0] for county in counties if county[0] not in exclude}
    return counties


def get_FRED_files(filename):
    """
    Parses FRED README file into list of lists, with each member list
    corresponding to an individual data series. Format of each member
    list follows that of README file itself, i.e:

    ['M\\N\\H\\MNHENN3POP.csv', 'Resident Population in Hennepin County,
     'MN', 'Thous. of Persons','A', 'NSA', '2015-08-05']

    Parameters
    ----------
    filename : string
        String indicating filename of README file provided within main
        FRED directory.

    Returns
    -------
    FRED_files : list
        List of lists. Each member list corresponds to a single data
        series, and has format described above.

    """
    with open(filename) as readme:
        FRED_files = [[item.strip().replace('\\', '/') for item in line.strip().split(";")] for line in readme.readlines() if ".csv" in line]
    return FRED_files


def get_PCPI_files(FRED_files, FIPS_L):
    """
    Returns list of files (with necessary paths) for PCPI data for each
    county in the specified FIPS list.
    
    Parameters
    ----------
    FRED_files : list
        List of FRED Files, obtained by get_FRED_files().
    FIPS_L : list
        List of FIPS codes for counties from which filenames are to be
        located.

    Returns
    -------
    Files_FIPS_L: list
        List with of tuples. Each tuple has format (filename, FIPS_code)

    Note: Since PCPI filenames contain FIPS codes, returned list is a
    little redundant. However, POP filenames do not contain the FIPS
    code, and thus will require returning both the filename and FIPS
    code. The redundancy is necessary to keep the code symmetric in this
    case.

    """
    Files_FIPS_L = [(fileI[0], fileI[0][-9:-4]) for fileI in FRED_files if re.match('PCPI\d{5}',fileI[0].split('.')[0][-9:]) and fileI[0][-9:-4] in FIPS_L]
    return Files_FIPS_L


def get_POP_files(FRED_files, FIPS_L, counties):
    """
    Returns list of files (with necessary paths) for POP data for each
    county in the specified FIPS list.
    
    Parameters
    ----------
    FRED_files : list
        List of FRED Files, obtained by get_FRED_files().
    FIPS_L : list
        List of FIPS codes for counties from which filenames are to be
        located.
    counties : dictionary
        Dictionary of format that matches the return value of
        get_county_dict, i.e. {FIPS code : Name of County, ST}

    Returns
    -------
    Files_FIPS_L: list
        List with of tuples. Each tuple has format (filename, FIPS_code)

    """
    prefix = "Resident Population in "
    FRED_files = [fileI for fileI in FRED_files if prefix in fileI[1]]
    titles = {prefix+counties[FIP] : FIP for FIP in FIPS_L}
    Files_FIPS_L = [(fileI[0], titles[fileI[1]]) for fileI in FRED_files if fileI[1] in titles]
    return Files_FIPS_L


def aggregate(files, label, save='pickle', verbose=True, sort=True):
    """
    Extracts data from each file in 'files' list from FRED data
    directory .csv files. Returns aggregated dataframe and list of FIPS
    that were successfully loaded (as new attribute of dataframe).
    
    Parameters
    ----------
    files : list
        List of FRED Files, obtained by get_PCPI_files() or
        get_POP_files().
    label : string
        Label to be appended to FIPS code to form column name in 
        aggregate dataframe, and appended to the title of saved files.
    save : string, optional
        If 'pickle' (default), aggregated dataframe will be saved as
        pickle (in current directory). If 'csv', dataframe will be saved
        as .csv. Any other value will will forego saving the aggregated
        file.
    verbose : bool, optional
        If True, prints to console upon access of each .csv.
    sort : bool, optional
        If True, columns will be ordered lexographically.

    Returns
    -------
    agg : pandas DataFrame object
        Aggregated data, as pandas DataFrame. Will have columns 'DATE' 
        (index), '27053_PCPI', '27041_PCPI', etc. List of loaded FIPS is
        appended as new attribute 'loaded'

    Note: Of 3235 counties or equivalents (2010 Census Data), 3082 have 
    PCPI data (and 3064 of these are complete on the period 1970 - 
    2013). 3123 have valid POP files, and the intersection of these sets
    has 3049 members.

    """
    
    agg = pd.DataFrame({}, index=dates_master)
    agg.index.name = 'DATE'
    num_files = str(len(files))
    str_width = len(num_files)
    
    for i, tup in enumerate(files):
        incoming, FIPS = tup # chang this to i, (incoming, fips)
        try:
            to_add = pd.read_csv(FRED_dir+'data/'+incoming, names=['DATE', FIPS+'_'+label], header=0, parse_dates=[0], index_col='DATE')
            if not np.all(dates_master.isin(to_add.index)):
                raise DateError
        except (IOError, DateError) as caught:
            if isinstance(caught, IOError) and verbose:
                print incoming+' does not exist! Skipping...'
            elif isinstance(caught, DateError) and verbose:
                print incoming+' is not complete! Skipping...'
            continue
        agg = agg.join(to_add)
        if verbose:
            print incoming+" (file {index:{width}d}".format(index=i+1, width=str_width)+" of "+ num_files+")"
    
    if sort:
        agg.sort_index(axis=1, inplace=True)
    
    if save == 'pickle':
        agg.to_pickle(label+'_pickle')
    elif save == 'csv':
        agg.to_csv(label+'_agg.csv')

    agg.loaded = [col[:5] for col in agg.columns.tolist()]
    agg.label = label
    return agg


def deflate(df, norm_filename='C/P/I/CPIAUCSL.csv', base='2015'):
    """
    Modifies monetary value of input dataframe (with annual frequency)
    using specified basis and base year. Example:

    df = deflate(df, 'C/P/I/CPIAUCSL.csv', '2015')
    will adjust the values of df on a year by year basis, such that the
    value at each year is normalized to the corresponding value in 2015
    dollars. CPIAUCSL is the Consumer Price Index, which is a commonly
    used basis for adjusting the value of USD between years in general.
    To adjust the GDP, it is preferable to use GDP deflators
    (F/Y/FYGDP.csv). Note that the normalization basis is itself
    normalized (by dividing by value at base year), and therefore the
    units of the normalization basis are eliminated.

    Parameters
    ----------
    df : pandas DataFrame object
        Dataframe to be normalized. Must be in 'AS' frequency.
    norm_filename : string, optional
        Filename of FRED data to be used as normalization basis. Default
        is Consumer Price Index. Basis must have values for the entire
        range of df.
    base : string, optional
        Year to normalize to. Default is 2015. Note that base need not
        be in the range of df's index.

    Returns
    -------
    df_norm : pandas DataFrame object
        Dataframe with values adjusted, such that effects of inflation
        are eliminated.

    """
    norm    = pd.read_csv(FRED_dir+'data/'+norm_filename, header=0, index_col='DATE', parse_dates=[0])
    norm    = norm.resample('AS')
    norm   /= norm.loc[base]['VALUE'][0]
    norm    = norm[norm.index.isin(df.index)]
    df_norm = pd.DataFrame(df.values/norm.values, columns=df.columns, index=df.index)
    return df_norm


def interp_data(df, intervals=2, kind='cubic'):
    """
    Interpolates data using dataframe's interpolate() method. Each year
    will expand to intervals number of measurements. Example:

    df = interp_data(df, intervals=2)
    will result in measurements at 1970.0, 1970.5, 1980.0, etc...


    Parameters
    ----------
    df : pandas DataFrame object
        Dataframe to be interpolated. Must be in 'AS' frequency.
    intervals : int, optional
        Number of intervals to interpolate between years.
    kind : string, optional
        Interpolation method. Default is 'cubic', which is cubic spline
        interpolation. Linear interpolation can be specified with
        'linear'. Consult pandas documentation for other options.

    Returns
    -------
    df_out : pandas DataFrame object
        Dataframe with interpolated values and non-datetime index.
    """
    delta        = 1.0 / intervals
    base_index   = df.index.year
    df_out       = df.copy(deep=True)
    df_out.index = base_index
    new_index    = np.arange(base_index.min(), base_index.max()+delta, delta)
    df_out       = df_out.reindex(new_index, tolerance=delta*0.1, method='nearest')
    df_out.interpolate(kind, inplace=True)
    return df_out


def make_kdes(x_data, y_data, lims=[], N=100, foldername='kde_plots', pkg='sp'):
    """
    Calculates KDE's of x_data vs y_data, and saves each timepoint as
    .png files in the specified folder (which will be created if it
    does not exist, and otherwise overwritten). Uses multiple processes
    as this can be computationally heavy. This process takes a few
    minutes for uninterpolated data, but can take considerably longer
    if interpolation has been used. For 15x interpolation, the rendering
    times on my machine were about an hour. During this time most of
    your systems CPU and memory will be tied up, so plan accordingly.

    Parameters
    ----------
    x_data : pandas DataFrame object
        Dataframe that represents x data. Index must be numerical (i.e.
        not datetime)
    y_data : pandas DataFrame object
        Dataframe that represents y data. Index must be numerical (i.e.
        not datetime)
    lims : list
        List of values for axes limits, of form
        x_min, x_max, x_scale (major scale tick), y_min, y_max, y_scale
    N : int
        Number of points in grid for KDE evaluation. For best results,
        use at least 400. Default value (100) produces rough graph.
    foldername : string
        Name of folder in which frames will be deposited. Folder will
        either be created (if it does not exist) or overwritten.
    pkg : string
        Either 'sp' for scipy or 'sk' for scikit-learn. The main
        difference in these results (as currently written) is that 'sk'
        will employ Euclidean distance metric for calculating kernels
        (i.e. perfectly round contours for single points) and 'sp' will
        use Mahalanobis distance. 'sk' should only be used if the x and
        y axes have the same scale.


    Returns
    -------
    None
    """
    current_dir = os.getcwd()

    try:
        os.mkdir(foldername)
    except OSError:
        shutil.rmtree(foldername) # clear it out if it already exists
        os.mkdir(foldername)
    finally:
        os.chdir(foldername)

    if not lims:
        lims = [x_data.min(), x_data.max(), 20, y_data.min(), y_data.max(), 1]

    mpl.rcParams['axes.linewidth'] = .5
    mpl.rcParams['lines.linewidth'] = .5
    n_processes = mp.cpu_count() - 1
    pool = mp.Pool(processes = n_processes)
    n_plots = x_data.shape[0]
    width = len(str(n_plots))
    x_arr = x_data.values
    y_arr = y_data.values
    inps  = [np.vstack([x_i, y_i]) for x_i, y_i in zip(x_arr, y_arr)]

    extr  = _get_extrema(x_data, 5)
    lims  = [lims]*n_plots
    N     = [N]*n_plots
    pkg   = [pkg]*n_plots

    yr_index    = x_data.index.values
    delta       = np.diff(yr_index).mean()
    truncated   = np.trunc(yr_index)
    correction  = (yr_index - truncated) > (1 - delta) # boolean mask for correction necessary due to float accuracy issues
    
    years = truncated
    years[correction] += 1
    years = [int(year) for year in years]

    labels = ["{num:0{width}}".format(num=i, width=width) for i in range(n_plots)]
    args = zip(inps, years, extr, lims, N, labels, pkg)
    chunks = len(args) / n_processes
    for p in pool.imap_unordered(_kde_drone, args, chunks):
        p
        gc.collect() # necessary to solve memory leak

    os.chdir(current_dir)


def make_z_kdes(x_data, y_data, lims=6, N=100, foldername='z_kde_plots', pkg='sp'):
    """
    Calculates KDE's of standard ('z') scores in x_data vs y_data, and
    saves each timepoint as .png files in the specified folder (which
    will be created if it does not exist, and otherwise overwritten).
    Uses multiple processes as this can be computationally heavy.
    This process takes a few minutes for uninterpolated data, but can
    take considerably longer if interpolation has been used. For 15x
    interpolation, the rendering times on my machine were about an hour.
    During this time most of your systems CPU and memory will be tied
    up, so plan accordingly.

    Note that using standard scores obviates the need for adjusting for
    inflation. However, the adjustment will also not affect the end
    result.

    Parameters
    ----------
    x_data : pandas DataFrame object
        Dataframe that represents x data. Index must be numerical (i.e.
        not datetime)
    y_data : pandas DataFrame object
        Dataframe that represents y data. Index must be numerical (i.e.
        not datetime)
    lims : list
        List of values for axes limits, of form
        x_min, x_max, x_scale (major scale tick), y_min, y_max, y_scale
    N : int
        Number of points in grid for KDE evaluation. For best results,
        use at least 400. Default value (100) produces rough graph.
    foldername : string
        Name of folder in which frames will be deposited. Folder will
        either be created (if it does not exist) or overwritten.
    pkg : string
        Either 'sp' for scipy or 'sk' for scikit-learn. The main
        difference in these results (as currently written) is that 'sk'
        will employ Euclidean distance metric for calculating kernels
        (i.e. perfectly round contours for single points) and 'sp' will
        use Mahalanobis distance. 'sk' should only be used if the x and
        y axes have the same scale.


    Returns
    -------
    None
    """

    x_data = x_data.apply(stats.zscore, axis=1)
    y_data = y_data.apply(stats.zscore, axis=1)

    current_dir = os.getcwd()

    try:
        os.mkdir(foldername)
    except OSError:
        shutil.rmtree(foldername) # clear it out if it already exists
        os.mkdir(foldername)
    finally:
        os.chdir(foldername)

    mpl.rcParams['axes.linewidth'] = .5
    mpl.rcParams['lines.linewidth'] = .5
    n_processes = mp.cpu_count() - 1
    pool = mp.Pool(processes = n_processes)
    n_plots = x_data.shape[0]
    width = len(str(n_plots))
    x_arr = x_data.values
    y_arr = y_data.values
    inps  = [np.vstack([x_i, y_i]) for x_i, y_i in zip(x_arr, y_arr)]

    lims  = [lims]*n_plots
    N     = [N]*n_plots
    pkg   = [pkg]*n_plots

    yr_index    = x_data.index.values
    delta       = np.diff(yr_index).mean()
    truncated   = np.trunc(yr_index)
    correction  = (yr_index - truncated) > (1 - delta) # boolean mask for correction necessary due to float accuracy issues
    
    years = truncated
    years[correction] += 1
    years = [int(year) for year in years]

    labels = ["{num:0{width}}".format(num=i, width=width) for i in range(n_plots)]
    args = zip(inps, years, lims, N, labels, pkg)
    chunks = len(args) / n_processes
    for p in pool.imap_unordered(_z_kde_drone, args, chunks):
        p
        gc.collect() # necessary to solve memory leak

    os.chdir(current_dir)


def _kde_drone(args):
    """
    Drone function to be called by multiprocessing Pool in make_kdes.
    Not intended to be called manually.
    """
    data, year, extr, lims, N, label, pkg = args
    xmin, xmax, x_majscale, ymin, ymax, y_majscale = lims
    top_vals, top_names, bot_vals, bot_names = extr

    w = 5
    h = 3

    kde = stats.gaussian_kde(data, bw_method='silverman')
    kde.set_bandwidth(kde.factor / 2)
    base_level = 1/(2*np.pi*np.linalg.det(kde.covariance)**0.5)/data.shape[1]*0.6
    levels = [base_level*2**i for i in range(9)]
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    positions = np.vstack([xx.ravel(), yy.ravel()])

    if pkg == 'sp':
        zz = kde(positions).T.reshape(xx.shape)
    elif pkg == 'sk':
        kde = KernelDensity(kde.factor)
        kde.fit(data.T)
        zz = np.exp(kde.score_samples(positions.T)).reshape(xx.shape)
    else:
        raise ValueError("""pkg must be either 'sk' or 'sp'""")

    fig, ax = plt.subplots(figsize=(w, h))
    
    pardir   = os.path.abspath(os.pardir)
    fontfile = pardir+'/Futura-Medium.ttf' # change as needed
    if os.path.isfile(fontfile):
        props = fm.FontProperties(fname=fontfile, size=6)
        propl = fm.FontProperties(fname=fontfile, size=12)
    else:
        props = fm.FontProperties(family='sans-serif', size=6)
        propl = fm.FontProperties(family='sans-serif', size=12)
    
    # set up axes
    plt.ylabel(r'County Population', fontproperties=props)
    plt.xlabel(r'Per Capita Personal Income (2015 Dollars, Thousands)', fontproperties=props)
    xmajorLocator = MultipleLocator(x_majscale)
    xminorLocator = MultipleLocator(x_majscale/2)
    ymajorLocator = FixedLocator(range(ymin, ymax+1, y_majscale)) # this was necessary (over MultipleLocator) for correct tick labels
    yminorLocator = FixedLocator([j+np.log10(i) for j in range(ymin,ymax) for i in range(1,10)])
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.set_yticklabels(['$\mathregular{{10^{num}}}$'.format(num=i) for i in ax.get_yticks()], minor=False, fontproperties=props)
    for label_i in ax.get_xticklabels():
        label_i.set_fontproperties(props)

    # draw contour plot
    cmap = ListedColormap(Spectral9)
    ax.contourf(xx, yy, zz, cmap=cmap, levels=levels, norm=LogNorm())
    ax.contour(xx, yy, zz, colors='k', levels=levels)

    # add year to corner
    fig.text(0.775,0.8,year, transform=ax.transAxes, fontproperties=propl, horizontalalignment='left', verticalalignment='baseline')

    # add manual color scale
    r = 0.012
    colors = list(Spectral9) # need to copy list before modifying, or later processes will have truncated list
    del colors[3]
    cs_starty  = 0.2
    cs_endy    = 0.8
    cs_tscale  = 0.8 # bug with placing text, axes transform does not work (always in figure mode)
    cs_tboosty = 0.12
    cs_cx = 1.055
    cs_tx = 0.97
    delta = (cs_endy - cs_starty) / (len(colors)-1)
    for i, c in enumerate(colors):
        fig.text(cs_tx, cs_tscale*(cs_starty+delta*i+cs_tboosty), 2**i, transform=ax.transAxes, fontproperties=props, horizontalalignment='left', verticalalignment='center')
        c = mpl.patches.Ellipse(xy=(cs_cx,cs_starty+delta*i), width=h*r, height=w*r, transform=ax.transAxes, linewidth=0.5, facecolor=c, edgecolor='k', clip_on=False)
        ax.add_artist(c)

    # save version with minimal annotations
    fig.savefig('min_'+label+'.png', dpi=300, bbox_inches='tight')

    # add statistics
    display_str = 'rho:\nmean PCPI:\nskewness:'
    measure_str = '{rho:+5.2f}\n{mean:5.2f}\n{skew:5.3f}'.format(rho=stats.spearmanr(data, axis=1)[0], mean=data[0].mean(), skew=stats.skew(data[0]))
    fig.text(0.23,0.81,display_str, transform=ax.transAxes, fontproperties=props, horizontalalignment='right', verticalalignment='center', linespacing=1.0)
    fig.text(0.24,0.81,measure_str, transform=ax.transAxes, fontproperties=props, horizontalalignment='left', verticalalignment='center', linespacing=1.0)

    # save version with basic annotations
    fig.savefig('med_'+label+'.png', dpi=300, bbox_inches='tight')

    # place lowest and highest values and text in lower part of figure
    txt_y = 0.365
    txt_x_lft = 0.19
    txt_x_rgt = 0.82
    delta = 0.0525
    vals_str  = '\n{: 4.1f}\n{: 4.1f}\n{: 4.1f}\n{: 4.1f}\n{: 4.1f}'.format(*top_vals)
    names_str = 'Top 5 PCPI\n{}\n{}\n{}\n{}\n{}'.format(*[name+':' for name in top_names])
    bbox_dict = {'pad' : 3, 'edgecolor' : (1, 1, 1, 0), 'facecolor' : (1, 1, 1, 0.7)}

    thv = fig.text(txt_x_rgt+0.01,txt_y-delta, vals_str, transform=ax.transAxes, fontproperties=props, horizontalalignment='left', verticalalignment='top', linespacing=1.0)
    thv.set_bbox(bbox_dict)
    thn = fig.text(txt_x_rgt,txt_y-delta, names_str, transform=ax.transAxes, fontproperties=props, horizontalalignment='right', verticalalignment='top', linespacing=1.0)
    thn.set_bbox(bbox_dict)

    vals_str  = '\n{:> 4.1f}\n{:> 4.1f}\n{:> 4.1f}\n{:> 4.1f}\n{:> 4.1f}'.format(*bot_vals)
    names_str = ' Bottom 5 PCPI\n{}\n{}\n{}\n{}\n{}'.format(*[' :'+name for name in bot_names])

    tln = fig.text(txt_x_lft,txt_y-delta, names_str, transform=ax.transAxes, fontproperties=props, horizontalalignment='left', verticalalignment='top', linespacing=1.0)
    tln.set_bbox(bbox_dict)
    tlv = fig.text(txt_x_lft-0.01,txt_y-delta, vals_str, transform=ax.transAxes, fontproperties=props, horizontalalignment='right', verticalalignment='top', linespacing=1.0)
    tlv.set_bbox(bbox_dict)

    # save version with full annotations
    fig.savefig('full_'+label+'.png', dpi=300, bbox_inches='tight')

    # free up memory
    del data
    del kde
    ax.cla()
    fig.clf()
    plt.close(fig)
    gc.collect()


def _z_kde_drone(args):
    """
    Drone function to be called by multiprocessing Pool in make_z_kdes.
    Not intended to be called manually.
    """
    data, year, lims, N, label, pkg = args

    w = 3
    h = 3

    kde = stats.gaussian_kde(data, bw_method='silverman')
    kde.set_bandwidth(kde.factor / 2)
    base_level = 1/(2*np.pi*np.linalg.det(kde.covariance)**0.5)/data.shape[1]*0.6
    levels = [base_level*2**i for i in range(9)]
    x = np.linspace(-lims, lims, N)
    y = np.linspace(-lims, lims, N)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    positions = np.vstack([xx.ravel(), yy.ravel()])

    if pkg == 'sp':
        zz = kde(positions).T.reshape(xx.shape)
    elif pkg == 'sk':
        kde = KernelDensity(kde.factor)
        kde.fit(data.T)
        zz = np.exp(kde.score_samples(positions.T)).reshape(xx.shape)
    else:
        raise ValueError("""pkg must be either 'sk' or 'sp'""")

    fig, ax = plt.subplots(figsize=(w, h))
    plt.subplots_adjust(0,0,0.8,0.8)

    pardir   = os.path.abspath(os.pardir)
    fontfile = pardir+'/Futura-Medium.ttf' # change as needed
    if os.path.isfile(fontfile):
        props = fm.FontProperties(fname=fontfile, size=6)
        propl = fm.FontProperties(fname=fontfile, size=12)
    else:
        props = fm.FontProperties(family='sans-serif', size=6)
        propl = fm.FontProperties(family='sans-serif', size=12)

    # set up axes
    plt.ylabel(r'County Population (standard score)', fontproperties=props)
    plt.xlabel(r'Per Capita Personal Income (standard score)', fontproperties=props)
    ax.set_xlim([-lims, lims])
    ax.set_ylim([-lims, lims])
    majorLocator = MultipleLocator(2)
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    for label_i in ax.get_xticklabels():
        label_i.set_fontproperties(props)
    for label_i in ax.get_yticklabels():
        label_i.set_fontproperties(props)
    
    # draw contour plot
    cmap = ListedColormap(Spectral9)
    ax.contourf(xx, yy, zz, cmap=cmap, levels=levels, norm=LogNorm())
    ax.contour(xx, yy, zz, colors='k', levels=levels)

    # add year to corner
    fig.text(0.591,0.7, year, transform=ax.transAxes, fontproperties=propl, horizontalalignment='left', verticalalignment='baseline')

    # add manual color scale
    r = 0.02
    colors = list(Spectral9)
    del colors[3]
    cs_starty  = 0.2
    cs_endy    = 0.8
    cs_tscale  = 0.8 # bug with placing text, axes transform does not work (always in figure mode)
    cs_tboosty = -0.005
    cs_cx = 1.09
    cs_tx = 0.918
    delta = (cs_endy - cs_starty) / (len(colors)-1)
    for i, c in enumerate(colors):
        fig.text(cs_tx, cs_tscale*(cs_starty+delta*i+cs_tboosty), 2**i, transform=ax.transAxes, fontproperties=props, horizontalalignment='left', verticalalignment='center')
        c = mpl.patches.Ellipse(xy=(cs_cx,cs_starty+delta*i), width=h*r, height=w*r, transform=ax.transAxes, linewidth=0.5, facecolor=c, edgecolor='k', clip_on=False)
        ax.add_artist(c)

    # add integer sd rings
    for j in range(1, lims):
        c = mpl.patches.Circle(xy=(0,0), radius=j, linewidth=0.5, facecolor='none', edgecolor='k', clip_on=True, zorder=10, alpha=0.5)
        ax.add_artist(c)

    # save png
    fig.savefig(label+'.png', dpi=300, bbox_inches='tight')

    # free up memory
    del data
    del kde
    ax.cla()
    fig.clf()
    plt.close(fig)
    gc.collect()    


def _get_extrema(df, num):
    """
    Returns num highest and lowest values (and county names) for df.
    Intended as helper function for make_kdes
    """
    counties_dict = get_county_dict(county_file)
    cols = df.columns
    out  = []
    for row in df.values:
        inds = np.argsort(row)
        top = inds[-num:]
        bot = inds[:num]
        top_vals  = row[top][::-1]
        top_names = [counties_dict[val[:5]] for val in cols[top]][::-1]
        bot_vals  = row[bot]
        bot_names = [counties_dict[val[:5]] for val in cols[bot]]
        out.append([top_vals, top_names, bot_vals, bot_names])
    return out


if __name__ == '__main__':

    # The FRED dataset (csv format) can be found here:
    # https://research.stlouisfed.org/fred2/downloaddata/

    # The list of counties can be found here:
    # http://www2.census.gov/geo/docs/reference/codes/files/national_county.txt

    # To get plots that are identical to those presented, you will have
    # to locate a Futura-Medium.ttf file and place it in the directory
    # that you call this script from

    # Adjust as needed
    FRED_dir    = '/Users/tbk/FRED2_csv_2/'
    county_file = '/Users/tbk/national_county.txt'

    # First step: aggregate raw FRED PCPI data into a single dataframe
    # Could do POP first with about equal performance (though 3082 of
    # the 3143 counties have valid PCPI data, while 3123 have valid POP
    # data, thus you'll end up loading and throwing out more unnecessary
    # data).
    counties   = get_county_dict(county_file)
    FRED_files = get_FRED_files(FRED_dir+'README_SERIES_ID_SORT.txt')
    PCPI_files = get_PCPI_files(FRED_files, counties.keys())
    PCPI_data  = aggregate(PCPI_files, 'PCPI')

    # Then, load POP data for counties which have PCPI data
    POP_files  = get_POP_files(FRED_files, PCPI_data.loaded, counties)
    POP_data   = aggregate(POP_files, 'POP')

    # Remove data from PCPI for counties that did not have POP data
    to_remove  = [PCPI_FIP+'_'+PCPI_data.label for PCPI_FIP in PCPI_data.loaded if PCPI_FIP not in POP_data.loaded]
    PCPI_data.drop(to_remove, axis=1, inplace=True)



    # Adjust PCPI data to 2015 dollars, and convert to thousands of
    # dollars
    PCPI_data  = deflate(PCPI_data, 'C/P/I/CPIAUCSL.csv', base='2015')
    PCPI_data /= 1e3

    # Convert POP data from thousands, and take log10
    POP_data  *= 1e3
    POP_data   = POP_data.apply(np.log10)

    # Interpolate to desired amount
    # n_intervals is total number of frames per year (including 
    # original data). E.g. for n_intervals = 15 means 15 frames per year
    # Even if you don't intend to interpolate, downstream code will
    # require the reindexing to numerical units. In this case, just set
    # n_intervals to 1. The difference between the result and the actual
    # data in this case is typically less than 1 part per trillion
    n_intervals = 1 # for 30 fps at 2 years per second, set to 15. WARNING: This will take a long time to render
    POP_data  = interp_data(POP_data, n_intervals)
    PCPI_data = interp_data(PCPI_data, n_intervals)



    # Calculate KDE's and render plots as pngs
    make_kdes(PCPI_data, POP_data, [0, 100, 20, 1, 8, 1], 100)
    make_z_kdes(PCPI_data, POP_data, 6, 100) 









