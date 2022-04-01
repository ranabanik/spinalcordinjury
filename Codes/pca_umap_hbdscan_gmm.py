import os
from glob import glob
import numpy as np
from numpy.lib.stride_tricks import as_strided
from imzml import IMZMLExtract, Binning2, normSpec, normalize_spectrum
from scipy.io import savemat, loadmat
import scipy.cluster.hierarchy as shc
from scipy import ndimage, signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
import hdbscan
from sklearn.mixture import GaussianMixture as GMM
import matplotlib as mtl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.express as px
import seaborn as sns
import pywt
import pandas as pd
import time
import copy
# from matplotlib import widget
from IPython import get_ipython
# %matplotlib inline
# import matplotlib
mtl.use('TkAgg')
# mtl.use('GTK3Agg')


# get_ipython().run_line_magic('matplotlib', 'inline')

exprun_name = 'pca_umap_hdbscan_gmm_'
TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
if exprun_name:
    exprun = exprun_name + TIME_STAMP
else:
    exprun = TIME_STAMP

print(exprun)

RandomState = 20210131

oldCDir = r'C:\Data\PosLip'

def printStat(data):
    data = np.array(data)
    return print("Max:{:.4f},  Min:{:.4f},  Mean:{:.4f},  Std:{:.4f}".format(np.max(data), np.min(data), np.mean(data), np.std(data)))
def makeSS(x):
    """
    x : the signal/data to standardized
    return : signal/data with zero mean, unit variance
    """
    u = np.mean(x)
    s = np.std(x)
    assert (s != 0)
    return (x - u)/s
def find_nearest(array, value):
    """
    locate the nearest element from an array with respect to a specific value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def chart(X, y):
    # --------------------------------------------------------------------------#
    # This section is not mandatory as its purpose is to sort the data by label
    # so, we can maintain consistent colors for digits across multiple graphs

    # Concatenate X and y arrays
    arr_concat = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    # --------------------------------------------------------------------------#

    # Create a 3D graph
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), height=900, width=950)

    # Update chart looks
    fig.update_layout(title_text='UMAP',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    fig.show()
def _generate_nComponentList(n_class, span):
    n_component = np.linspace(n_class-int(span/2), n_class+int(span/2), span).astype(int)
    return n_component
def retrace_columns(df_columns, keyword): # df_columns: nparray of df_columns. keyword: str
    counts = 0
    for i in df_columns:
        element = i.split('_')
        for j in element:
            if j == keyword:
                counts += 1
    return counts
def nnPixelCorrect(arr, n_, d, bg_=0, plot_=True):
    """
    corrects the pixel value based on neighnoring pixels
    n_: value of noise pixel to correct
    bg_: backgroud pixel value, default 0.
    d: degree of neighbor
    """
    def sliding_window(arr, window_size):
        """ Construct a sliding window view of the array"""
        arr = np.asarray(arr)
        window_size = int(window_size)
        if arr.ndim != 2:
            raise ValueError("need 2-D input")
        if not (window_size > 0):
            raise ValueError("need a positive window size")
        shape = (arr.shape[0] - window_size + 1,
                 arr.shape[1] - window_size + 1,
                 window_size, window_size)
        if shape[0] <= 0:
            shape = (1, shape[1], arr.shape[0], shape[3])
        if shape[1] <= 0:
            shape = (shape[0], 1, shape[2], arr.shape[1])
        strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
                   arr.shape[1]*arr.itemsize, arr.itemsize)
        return as_strided(arr, shape=shape, strides=strides)

    def cell_neighbors(arr, i, j, d):
        """Return d-th neighbors of cell (i, j)"""
        w = sliding_window(arr, 2*d+1)
        ix = np.clip(i - d, 0, w.shape[0]-1)
        jx = np.clip(j - d, 0, w.shape[1]-1)
        i0 = max(0, i - d - ix)
        j0 = max(0, j - d - jx)
        i1 = w.shape[2] - max(0, d - i + ix)
        j1 = w.shape[3] - max(0, d - j + jx)
        return w[ix, jx][i0:i1, j0:j1].ravel()

    def most_common(lst):
        return max(set(lst), key=lst.count)
    arr_ = copy.deepcopy(arr)
    noiseIndices = np.where(arr == n_)
    listOfCoordinates = list(zip(noiseIndices[0], noiseIndices[1]))
    for coOrd in listOfCoordinates:
        # print("noise ind: ", cord[0], cord[1])
        cn = cell_neighbors(arr, coOrd[0], coOrd[1], d)
        cn = np.delete(cn, np.where((cn == bg_) | (cn == n_)))
        arr[coOrd[0], coOrd[1]] = most_common(cn.tolist())
    if plot_:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        arrays = [arr_, arr]
        title = ['noisy', 'corrected']
        fig, axs = plt.subplots(1, 2, figsize=(10, 8), dpi=200, sharex=False)
        for ar, tl, ax in zip(arrays, title, axs.ravel()):
            im = ax.imshow(ar) #, cmap='twilight') #cm)
            ax.set_title(tl, fontsize=20)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, ax=ax)
        plt.show()
    return arr
def _smooth_spectrum(spectrum, method="savgol", window_length=5, polyorder=2):
    assert (method in ["savgol", "gaussian"])
    if method == "savgol":
        outspectrum = signal.savgol_filter(spectrum, window_length=window_length, polyorder=polyorder, mode='nearest')
    elif method == "gaussian":
        outspectrum = ndimage.gaussian_filter1d(spectrum, sigma=window_length, mode='nearest')
    outspectrum[outspectrum < 0] = 0
    return outspectrum

if __name__ !='__main__':
    ## load data
    oldCFile = glob(os.path.join(oldCDir, '*.imzML'))
    print(oldCFile[0])
    oldCImz = IMZMLExtract(oldCFile[0])
    ##
    regionID = 1
    oldLipidObj = Binning2(oldCImz, 1)
    oldLipid1 = oldLipidObj.getBinMat()
    nSpecs, nBins = oldLipid1[1].shape
    print("There are {} spectrums and {} m/z bins".format(nSpecs, nBins))
    matr = {"data": oldLipid1[1], "label": "region 1(only spectras)"}
    savemat(os.path.join(oldCDir, 'oldLipid1.mat'), matr)

## read data
if __name__ != '__main__':
    datapath = glob(os.path.join(oldCDir, 'oldLipid1.mat'))
    print(datapath[0])
    data_obj = loadmat(datapath[0])
    print(sorted(data_obj.keys()))

    data = data_obj['data']
    print(data.shape)
    nSpecs, nBins = data.shape
    printStat(data)

# +-------------------+
# |   standardize     |
# +-------------------+
if __name__ != '__main__':
    ## min max normalization
    # normScaler = MinMaxScaler()
    #     # data_norm = normScaler.fit_transform(data)
    #     # printStat(data_norm)
    # data_norm = data

    data_norm = np.zeros_like(data)
    for s in range(0, nSpecs):
        data_norm[s, :] = normalize_spectrum(data[s, :], normalize='tic')
    printStat(data_norm)

    ## standardization
    # stanScaler = StandardScaler(with_mean=True, with_std=True)
    # data_norm_ss = stanScaler.fit_transform(data_norm)
    data_norm_ss = makeSS(data_norm)
    print("Standardized stat: ")
    printStat(data_norm_ss)
    data = pd.DataFrame(data_norm_ss)
    pixel_feature = data.values.astype(np.float64)
    print("Pixel feature: ")
    printStat(pixel_feature)

# +------------------------------------------------+
# | how many clusters?                             |
# | >> according to the dendogram on norm_ss data  |
# | there are 3/4 clusters in region 1             |
# +------------------------------------------------+
if __name__ != '__main__':
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms: Positive Lipid 1(Min-Max scaled)")
    dend = shc.dendrogram(shc.linkage(data_norm_ss, method='ward'))
    plt.show()

# +---------------+
# |   tree plots  |
# +---------------+
# if __name__ != '__main__':
#     clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#     plt.show()
#     clusterer.condensed_tree_.plot()
#     plt.show()
#     clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#     plt.show()

def msmlfunc(mspath, regID, threshold, exprun_name=None):
    """
    mspath: path to .imzML file
    regID: region to be analysed
    threshold: how much variance needed for PCA
    exprun_name:
    """
    RandomState = 20210131
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
    plot_spec = True
    plot_pca = True
    plot_umap = True
    save_rseg = True
    if exprun_name:
        exprun = exprun_name + '_' + TIME_STAMP
    else:
        exprun = TIME_STAMP
    # +------------------------------------+
    # |     read data and save region      |
    # +------------------------------------+
    dirname = os.path.dirname(mspath)
    basename = os.path.basename(mspath)
    filename, ext = os.path.splitext(basename)
    regname = os.path.join(dirname, '{}_reg_{}.mat'.format(filename, regID))
    ImzObj = IMZMLExtract(mspath)
    if os.path.isfile(regname):
        matr = loadmat(regname)
        regSpec = matr['data']
        regArr = matr['array']
        spCoo = matr['coordinates']
    else:
        BinObj = Binning2(ImzObj, regID)
        regArr, regSpec, spCoo = BinObj.getBinMat()
        matr = {"data": regSpec, "array": regArr, "coordinates": spCoo, "info": "after peakpicking in Cardinal"}
        savemat(regname, matr)    # basename
    nSpecs, nBins = regSpec.shape
    print(">> There are {} spectrums and {} m/z bins".format(nSpecs, nBins))
    # +------------------------------+
    # |   normalize, standardize     |
    # +------------------------------+
    reg_norm = np.zeros_like(regSpec)
    for s in range(0, nSpecs):
        # reg_norm[s, :] = normalize_spectrum(regSpec[s, :], normalize='tic')
        reg_norm_ = _smooth_spectrum(regSpec[s, :], method='savgol', window_length=7, polyorder=2)
        reg_norm[s, :] = normalize_spectrum(reg_norm_, normalize='tic')

        # printStat(data_norm)
    reg_norm_ss = makeSS(reg_norm).astype(np.float64)
    # reg_norm_ss = StandardScaler().fit_transform(reg_norm)
    # +----------------+
    # |  plot spectra  |
    # +----------------+
    if plot_spec:
        nS = np.random.randint(nSpecs)
        fig, ax = plt.subplots(4, 1, figsize=(16, 10), dpi=200)
        ax[0].plot(regSpec[nS, :])
        plt.title("raw spectrum")
        ax[1].plot(reg_norm[nS, :])
        plt.title("'tic' norm")
        ax[2].plot(reg_norm_ss[nS, :])
        plt.title("standardized")
        ax[3].plot(np.mean(regSpec, axis=0))
        plt.title("mean spectra(region {})".format(regID))
        plt.suptitle("Processing comparison of Spec #{}".format(nS))
        plt.show()

    data = copy.deepcopy(reg_norm_ss)
    # +------------+
    # |    PCA     |
    # +------------+
    pca_all = PCA(random_state=RandomState)
    pcs_all = pca_all.fit_transform(data)
    # pcs_all=pca_all.fit_transform(oldLipid_mm_norm)
    pca_range = np.arange(1, pca_all.n_components_, 1)
    print(">> PCA: number of components #{}".format(pca_all.n_components_))
    # printStat(pcs_all)
    evr = pca_all.explained_variance_ratio_
    print(evr)
    evr_cumsum = np.cumsum(evr)
    print(evr_cumsum)
    # threshold = 0.95  # to choose PCA variance
    cut_evr = find_nearest(evr_cumsum, threshold)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print(">> Nearest variance to threshold {:.4f} explained by PCA components {}".format(cut_evr, nPCs))
    df_pca = pd.DataFrame(data=pcs_all[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])

    if plot_pca:
        MaxPCs = nPCs + 5
        fig, ax = plt.subplots(figsize=(20, 8), dpi=200)
        ax.bar(pca_range[0:MaxPCs], evr[0:MaxPCs] * 100, color="steelblue")
        ax.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
        ax.set_xlabel('Principal component number', fontsize=30)
        ax.set_ylabel('Percentage of \nvariance explained', fontsize=30)
        ax.set_ylim([-0.5, 100])
        ax.set_xlim([-0.5, MaxPCs])
        ax.grid("on")

        ax2 = ax.twinx()
        ax2.plot(pca_range[0:MaxPCs], evr_cumsum[0:MaxPCs] * 100, color="tomato", marker="D", ms=7)
        ax2.scatter(nPCs, cut_evr * 100, marker='*', s=500, facecolor='blue')
        ax2.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
        ax2.set_ylabel('Cumulative percentage', fontsize=30)
        ax2.set_ylim([-0.5, 100])

        # axis and tick theme
        ax.tick_params(axis="y", colors="steelblue")
        ax2.tick_params(axis="y", colors="tomato")
        ax.tick_params(size=10, color='black', labelsize=25)
        ax2.tick_params(size=10, color='black', labelsize=25)
        ax.tick_params(width=3)
        ax2.tick_params(width=3)

        ax = plt.gca()  # Get the current Axes instance

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)

        plt.suptitle("PCA performed with {} features".format(pca_all.n_features_), fontsize=30)
        plt.show()

        plt.figure(figsize=(12, 10), dpi=200)
        plt.scatter(df_pca.PC_1, df_pca.PC_2, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
        plt.xlabel('PC1 ({}%)'.format(round(evr[0] * 100, 2)), fontsize=30)
        plt.ylabel('PC2 ({}%)'.format(round(evr[1] * 100, 2)), fontsize=30)
        plt.tick_params(size=10, color='black')
        # tick and axis theme
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax = plt.gca()  # Get the current Axes instance
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2)
        plt.suptitle("PCA performed with {} features".format(pca_all.n_features_), fontsize=30)
        plt.show()

    # +------------------+
    # |      UMAP        |
    # +------------------+
    u_neigh = 12
    u_comp = 3
    u_min_dist = 0.025
    reducer = UMAP(n_neighbors=u_neigh,
                   # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                   n_components=u_comp,  # default 2, The dimension of the space to embed into.
                   metric='cosine',
                   # default 'euclidean', The metric to use to compute distances in high dimensional space.
                   n_epochs=1000,
                   # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
                   learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
                   init='spectral',
                   # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                   min_dist=u_min_dist,  # default 0.1, The effective minimum distance between embedded points.
                   spread=1.0,
                   # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                   low_memory=False,
                   # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                   set_op_mix_ratio=1.0,
                   # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                   local_connectivity=1,
                   # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                   repulsion_strength=1.0,
                   # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                   negative_sample_rate=5,
                   # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                   transform_queue_size=4.0,
                   # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                   a=None,
                   # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                   b=None,
                   # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                   random_state=RandomState,
                   # default: None, If int, random_state is the seed used by the random number generator;
                   metric_kwds=None,
                   # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                   angular_rp_forest=False,
                   # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                   target_n_neighbors=-1,
                   # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                   # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
                   # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                   # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                   transform_seed=42,
                   # default 42, Random seed used for the stochastic aspects of the transform operation.
                   verbose=True,  # default False, Controls verbosity of logging.
                   unique=False
                   # default False, Controls if the rows of your data should be uniqued before being embedded.
                   )
    data_umap = reducer.fit_transform(df_pca.values)  # on pca
    for i in range(reducer.n_components):
        df_pca.insert(df_pca.shape[1], column='umap_{}'.format(i + 1), value=data_umap[:, i])
    df_pca_umap = copy.deepcopy(df_pca)
    # +---------------+
    # |   UMAP plot   |
    # +---------------+
    if plot_umap:
        plt.figure(figsize=(12, 10))
        plt.scatter(data_umap[:, 0], data_umap[:, 1], facecolors='None', edgecolors=cm.tab20(10), alpha=0.5)
        plt.xlabel('UMAP1', fontsize=30)  # only difference part from last one
        plt.ylabel('UMAP2', fontsize=30)

        # theme
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(size=10, color='black')

        ax = plt.gca()  # Get the current Axes instance

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.show()

    # +-------------+
    # |   HDBSCAN   |
    # +-------------+
    HDBSCAN_soft = False
    min_cluster_size = 250
    min_samples = 30
    cluster_selection_method = 'eom'  # eom
    if HDBSCAN_soft:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_method=cluster_selection_method, prediction_data=True) \
            .fit(data_umap)
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        labels = np.argmax(soft_clusters, axis=1)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_method=cluster_selection_method).fit(data_umap)
        labels = clusterer.labels_

    # process HDBSCAN data
    n_clusters_est = np.max(labels) + 1
    if HDBSCAN_soft:
        title = 'estimated number of clusters: ' + str(n_clusters_est)
    else:
        labels[labels == -1] = 19
        title = 'estimated number of clusters: ' + str(n_clusters_est) + ', noise pixels are coded in cyan'
    # plot
    plt.figure(figsize=(12, 10))
    plt.scatter(data_umap[:, 0], data_umap[:, 1], facecolors='None', edgecolors=cm.tab20(labels), alpha=0.9)
    plt.xlabel('UMAP1', fontsize=30)  # only difference part from last one
    plt.ylabel('UMAP2', fontsize=30)

    # theme
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(size=10, color='black')
    plt.title(title, fontsize=20)

    ax = plt.gca()  # Get the current Axes instance

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.show()

    chart(data_umap, labels)
    plt.suptitle("n_neighbors={}, Cosine metric".format(u_neigh))
    # plt.show()

    df_pca_umap.insert(df_pca_umap.shape[1], column='hdbscan_labels', value=labels)
    df_pca_umap_hdbscan = copy.deepcopy(df_pca_umap)
    # savecsv = os.path.join(dirname, '{}_reg_{}_{}.csv'.format(filename, regID, exprun))
    # df_pca_umap_hdbscan.to_csv(savecsv, index=False, sep=',')

    # +--------------------------------------+
    # |   Segmentation on PCA+UMAP+HDBSCAN   |
    # +--------------------------------------+
    regInd = ImzObj.get_region_indices(regID)
    xr, yr, zr, _ = ImzObj.get_region_range(regID)
    xx, yy, _ = ImzObj.get_region_shape(regID)
    sarray = np.zeros([xx, yy])
    for idx, coord in enumerate(regInd):
        print(idx, coord, ImzObj.coord2index.get(coord))
        xpos = coord[0] - xr[0]
        ypos = coord[1] - yr[0]
        sarray[xpos, ypos] = labels[idx] + 1    # to avoid making 0 as bg
    sarray = nnPixelCorrect(sarray, 20, 3)  # noisy pixel is labeled as 19 by hdbscan
    fig, ax = plt.subplots(figsize=(6, 8))
    sarrayIm = ax.imshow(sarray)
    fig.colorbar(sarrayIm)
    ax.set_title('reg{}: HDBSCAN labeling'.format(regID), fontsize=15, loc='center')
    plt.show()
    if save_rseg:
        namepy = os.path.join(dirname, '{}_reg_{}_{}_hdbscan-label.npy'.format(filename, regID, exprun))
        np.save(namepy, sarray)

    # +-----------------+
    # |       GMM       |
    # +-----------------+
    n_components = 5
    span = 5
    n_component = _generate_nComponentList(n_components, span)
    repeat = 2

    for i in range(repeat):  # may repeat several times
        for j in range(n_component.shape[0]):  # ensemble with different n_component value
            StaTime = time.time()
            gmm = GMM(n_components=n_component[j], max_iter=5000)  # max_iter does matter, no random seed assigned
            labels = gmm.fit_predict(data_umap)     #todo data_umap
            # save data
            index = j + 1 + i * n_component.shape[0]
            title = 'gmm_' + str(index) + '_' + str(n_component[j]) + '_' + str(i)
            # df_pixel_label[title] = labels

            SpenTime = (time.time() - StaTime)

            # progressbar
            print('{}/{}, finish classifying {}, running time is: {} s'.format(index, repeat * span, title,
                                        round(SpenTime, 2)))
            df_pca_umap_hdbscan.insert(df_pca_umap_hdbscan.shape[1], column=title, value=labels)

    df_pca_umap_hdbscan_gmm = copy.deepcopy(df_pca_umap_hdbscan)
    savecsv = os.path.join(dirname, '{}_reg_{}_{}.csv'.format(filename, regID, exprun))     # todo
    df_pca_umap_hdbscan_gmm.to_csv(savecsv, index=False, sep=',')

    nGs = retrace_columns(df_pca_umap_hdbscan_gmm.columns.values, 'gmm')
    df_gmm_labels = df_pca_umap_hdbscan_gmm.iloc[:, -nGs:]
    # print("gmm label: ", nGs)

    for (columnName, columnData) in df_gmm_labels.iteritems():
        print('Column Name : ', columnName)
        print('Column Contents : ', columnData.values)
        regInd = ImzObj.get_region_indices(regID)
        xr, yr, zr, _ = ImzObj.get_region_range(regID)
        xx, yy, _ = ImzObj.get_region_shape(regID)
        sarray1 = np.zeros([xx, yy])
        for idx, coord in enumerate(regInd):
            xpos = coord[0] - xr[0]
            ypos = coord[1] - yr[0]
            sarray1[xpos, ypos] = columnData.values[idx] + 1
        fig, ax = plt.subplots(figsize=(6, 8))
        sarrayIm = ax.imshow(sarray1)
        fig.colorbar(sarrayIm)
        ax.set_title('reg{}: umap_{}'.format(regID, columnName), fontsize=15, loc='center')
        plt.show()
    return

# posLip = r'/media/banikr2/DATA/MALDI/210427_Chen_pos_lipid' #
posLip = r'/media/banikr2/DATA/MALDI/fromCardinal/PosLip'
mspath = glob(os.path.join(posLip, '*.imzML'))[0]
print(mspath)

# msmlfunc(mspath, regID=1, threshold=0.95, exprun_name='sav_golay_norm')   # todo: change values

imze = IMZMLExtract(mspath)
spectra0_orig = imze.get_region_array(3, makeNullLine=True)

# +---------------------------+
# |   demonstrate smoothing   |
# +---------------------------+
if __name__ != '__main__':
    nX = 90  # np.random.randint(spectra0_orig.shape[0])
    nY = 41  # np.random.randint(spectra0_orig.shape[1])
    print(nX, nY)
    refSpec = spectra0_orig[nX, nY, :]
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    p = ax.plot(refSpec)
    wl_ = 3  # todo: 5 is better
    po_ = 2
    outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=wl_, polyorder=po_)
    p, = ax.plot(outspectrum)
    plt.subplots_adjust(bottom=0.25)
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    win_len = Slider(ax_slide, 'window length', valmin=wl_, valmax=99, valinit=99, valstep=2)
    def update(val):
        current_v = int(win_len.val)
        outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=current_v, polyorder=po_)
        p.set_ydata(outspectrum)
        fig.canvas.draw()
    win_len.on_changed(update)
    plt.show()

# +------------------------------------------------+
# |     undecimated discreet wavelet denoising     |
# +------------------------------------------------+
if __name__ != '__main__':
    def madev(d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def wavelet_denoising(x, wavelet='bior4.4', level=2):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1/0.6745) * madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')

    # signal = data[nS, :]

    for wav in pywt.wavelist():
        print(wav)
        # try:
        #     filtered = wavelet_denoising(signal, wavelet=wav, level=1)
        # except:
        #     pass

    filtered = wavelet_denoising(signal, wavelet='bior4.4')
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Raw')
    plt.plot(filtered, label='Filtered')
    plt.legend()
    plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(20, 8), dpi=200)
    # plt.subplot(411)
    ax[0].plot(signal)
    plt.title("raw")
    # plt.subplot(412)
    ax[1].plot(filtered)
    plt.title("filtered")
    plt.show()
    print(np.min(filtered))

# +---------------------------+
# |      ms_peak_picker       |
# +---------------------------+
if __name__ != '__main__':
    import ms_peak_picker

    # ms_peak_picker.peak_picker(spectra0_orig)
    picking_method = "quadratic"
    mz_array = np.array([23, 34])
    intensity_array = np.array([230, 340])
    peak_list = ms_peak_picker.pick_peaks(mz_array, intensity_array, fit_type=picking_method)

##
    nX = 90  # np.random.randint(spectra0_orig.shape[0])
    nY = 41  # np.random.randint(spectra0_orig.shape[1])
    print(nX, nY)
    refSpec = spectra0_orig[nX, nY, :][200:250]
    # smoothing signal before finding peaks
    wl_ = 5
    po_ = 2
    outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=wl_, polyorder=po_)
    peaks, properties = signal.find_peaks(outspectrum, prominence=1e6)
    print(peaks.shape, properties)
    # plt.plot(refSpec)
    plt.plot(outspectrum)
    plt.plot(peaks, outspectrum[peaks], "x")
    # plt.vlines(x=peaks, ymin=outspectrum[peaks] - properties["prominences"],
    #            ymax=outspectrum[peaks], color="C1")
    # plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
    #            xmax=properties["right_ips"], color="C1")
    plt.show()

class MaldiTofSpectrum(np.ndarray):
    """Numpy NDArray subclass representing a MALDI-TOF Spectrum."""
    def __new__(cls, peaks):
        """Create a MaldiTofSpectrum.
        Args:
            peaks: 2d array or list of tuples or list of list containing pairs
                of mass/charge to intensity.
        Raises:
            ValueError: If the input data is not in the correct format.
        """
        peaks = np.asarray(peaks).view(cls)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            raise ValueError(
                f'Input shape of {peaks.shape} does not match expected shape '
                'for spectrum [n_peaks, 2].'
            )
        return peaks

    @property
    def n_peaks(self):
        """Get number of peaks of the spectrum."""
        return self.shape[0]

    @property
    def intensities(self):
        """Get the intensities of the spectrum."""
        return self[:, 1]

    @property
    def mass_to_charge_ratios(self):
        """Get mass-t0-charge ratios of spectrum."""
        return self[:, 0]

MOCK_DATA = [
    MaldiTofSpectrum(
        [[0.0,   5.0],
         [10.7,  8.0],
         [150.4, 10.],
         [1000,  3.0]
         ]
    ),  # Mean intensity 6.5
    MaldiTofSpectrum(
        [[0.0,   15.0],
         [10.7,  0.0],
         [150.4, 10.],
         [1000,  3.0]
         ]
    ),  # Mean intensity 7 or 9.3333 (with ignore zero intensity)
]
print(MOCK_DATA[0].shape, MOCK_DATA[0].n_peaks, MOCK_DATA[0].mass_to_charge_ratios)

# MD = MaldiTofSpectrum(MOCK_DATA)
# print(MD)