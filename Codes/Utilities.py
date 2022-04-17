import os
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import copy
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat, savemat
from scipy import ndimage, signal
from scipy.spatial.distance import cosine
import pywt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from umap import UMAP
import hdbscan
import matplotlib as mtl
import matplotlib.cm as cm
import seaborn as sns
from imzml import IMZMLExtract, normalize_spectrum
from matchms import Spectrum, calculate_scores
from matchms.similarity import CosineGreedy, CosineHungarian, ModifiedCosine
from tqdm import tqdm

class Binning2(object):
    """
    given the imze object should create 3D matrix(spatial based) or 2D(spectrum based)
    spectrum: array with 2 vectors, one of abundance(1), other with m/z values(0)
    n_bins: number of bins/samples to be digitized
    plotspec: to plot the new binned spectrum, default--> True
    """
    def __init__(self, imzObj, regionID, plotspec=False):
        self.imzObj = imzObj
        self.regionID = regionID
        self.n_bins = len(imzObj.mzValues) + 1
        self.plotspec = plotspec
        self.xr, self.yr, self.zr, _ = self.imzObj.get_region_range(regionID)
        self.imzeShape = [self.xr[1] - self.xr[0] + 1,
                          self.yr[1] - self.yr[0] + 1, self.n_bins - 1]

    def getBinMat(self):
        sarray = np.zeros(self.imzeShape, dtype=np.float32)
        regInd = self.imzObj.get_region_indices(self.regionID)
        binned_mat = np.zeros([len(regInd), self.n_bins - 1])
        coordList = []
        #         xr, yr, zr, _ = self.imzObj.get_region_range(regionID)
        #         self.imzeShape = [xr[1]-xr[0]+1,
        #                  yr[1]-yr[0]+1, self.n_bins -1]
        for i, coord in enumerate(regInd):
            spectrum = self.imzObj.parser.getspectrum(self.imzObj.coord2index.get(coord))  # [0]
            bSpec = self.onebinning(spectrum)
            binned_mat[i] = bSpec
            xpos = coord[0] - self.xr[0]
            ypos = coord[1] - self.yr[0]
            sarray[xpos, ypos, :] = bSpec
            # coordList.append(coord)       # TODO: global
            coordList.append((xpos, ypos))    # local coordinates
        return sarray, binned_mat, coordList

    def onebinning(self, spectrum):
        """
        returns: binned_spectrum
        """
        bins = np.linspace(spectrum[0][0], spectrum[0][-1], num=self.n_bins, endpoint=True)
        hist = np.histogram(spectrum[0], bins=bins)
        binned_spectrum = np.zeros_like(hist[0])
        hstart = 0
        for i in range(len(hist[0])):
            binned_spectrum[i] = np.sum(spectrum[1][hstart:hstart + hist[0][i]])
            hstart = hstart + hist[0][i]
        if self.plotspec:
            plt.plot(bins[1:], binned_spectrum)
            plt.show()
        return binned_spectrum

def printStat(data):
    data = np.array(data)
    return print("Max:{:.4f},  Min:{:.4f},  Mean:{:.4f},  Std:{:.4f}".format(np.max(data), np.min(data), np.mean(data), np.std(data)))

def retrace_columns(df_columns, keyword): # df_columns: nparray of df_columns. keyword: str
    counts = 0
    for i in df_columns:
        element = i.split('_')
        for j in element:
            if j == keyword:
                counts += 1
    return counts

def _generate_nComponentList(n_class, span):
    n_component = np.linspace(n_class-int(span/2), n_class+int(span/2), span).astype(int)
    return n_component

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

def find_nearest(array, value):
    """
    locate the nearest element from an array with respect to a specific value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def makeSS(x):
    """
    x : the signal/data to standardized
    return : signal/data with zero mean, unit variance
    """
    u = np.mean(x)
    s = np.std(x)
    assert (s != 0)
    return (x - u)/s

def _smooth_spectrum(spectrum, method="savgol", window_length=5, polyorder=2):
    assert (method in ["savgol", "gaussian"])
    if method == "savgol":
        outspectrum = signal.savgol_filter(spectrum, window_length=window_length, polyorder=polyorder, mode='nearest')
    elif method == "gaussian":
        outspectrum = ndimage.gaussian_filter1d(spectrum, sigma=window_length, mode='nearest')
    outspectrum[outspectrum < 0] = 0
    return outspectrum

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

def msmlfunc(mspath, regID, threshold, exprun=None):
    """
    mspath: path to .imzML file
    regID: region to be analysed
    threshold: how much variance needed for PCA
    exprun: name of the experimental run

    e.g.: msmlfunc(mspath, regID=3, threshold=0.95, exprun='sav_golay_norm')
    """
    RandomState = 20210131
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
    plot_spec = True
    plot_pca = True
    plot_umap = True
    save_rseg = True
    # +------------------------------------+
    # |     read data and save region      |
    # +------------------------------------+
    dirname = os.path.dirname(mspath)
    basename = os.path.basename(mspath)
    filename, ext = os.path.splitext(basename)
    regDir = os.path.join(dirname, 'reg_{}'.format(regID))
    if not os.path.isdir(regDir):
        os.mkdir(regDir)
    regname = os.path.join(regDir, '{}_reg_{}.mat'.format(filename, regID))
    ImzObj = IMZMLExtract(mspath)
    if os.path.isfile(regname):
        matr = loadmat(regname)
        regArr = matr['array']
        regSpec = matr['spectra']
        spCoo = matr['coordinates']
    else:
        BinObj = Binning2(ImzObj, regID)
        regArr, regSpec, spCoo = BinObj.getBinMat()
        matr = {"spectra": regSpec, "array": regArr, "coordinates": spCoo, "info": "after peakpicking in Cardinal"}
        savemat(regname, matr)    # basename
    nSpecs, nBins = regSpec.shape
    print(">> There are {} spectrums and {} m/z bins".format(nSpecs, nBins))
    # +------------------------------+
    # |   normalize, standardize     |
    # +------------------------------+
    reg_norm = np.zeros_like(regSpec)
    _, reg_smooth_ = bestWvltForRegion(regSpec, bestWvlt='db8', smoothed_array=True, plot_fig=False)
    for s in range(0, nSpecs):
        reg_norm[s, :] = normalize_spectrum(reg_smooth_[s, :], normalize='tic')
        # reg_norm_ = _smooth_spectrum(regSpec[s, :], method='savgol', window_length=5, polyorder=2)
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
    # savecsv = os.path.join(regDir, '{}_{}.csv'.format(filename, exprun))
    # df_pca_umap_hdbscan.to_csv(savecsv, index=False, sep=',')

    # +--------------------------------------+
    # |   Segmentation on PCA+UMAP+HDBSCAN   |
    # +--------------------------------------+
    regInd = ImzObj.get_region_indices(regID)
    xr, yr, zr, _ = ImzObj.get_region_range(regID)
    xx, yy, _ = ImzObj.get_region_shape(regID)
    sarray = np.zeros([xx, yy])
    for idx, coord in enumerate(regInd):
        # print(idx, coord, ImzObj.coord2index.get(coord))
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
        namepy = os.path.join(regDir, '{}_hdbscan-label.npy'.format(exprun))
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
    savecsv = os.path.join(regDir, '{}.csv'.format(exprun))
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
        if save_rseg:
            namepy = os.path.join(regDir, 'umap-{}_{}.npy'.format(exprun, columnName))
            np.save(namepy, sarray)
    return

def msmlfunc2(dirname, regArr, regSpec, regCoor, regID, threshold, exprun_name=None):
    """
    mspath: path to .imzML file
    regID: region to be analysed
    threshold: how much variance needed for PCA
    exprun_name:

    e.g.: msmlfunc(mspath, regID=3, threshold=0.95, exprun_name='sav_golay_norm')
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
    regDir = os.path.join(dirname, 'reg_{}'.format(regID))
    if not os.path.isdir(regDir):
        os.mkdir(regDir)
    nSpecs, nBins = regSpec.shape
    print(">> There are {} spectrums and {} m/z bins".format(nSpecs, nBins))
    # +------------------------------+
    # |   normalize, standardize     |
    # +------------------------------+
    reg_norm = np.zeros_like(regSpec)
    _, reg_smooth_ = bestWvltForRegion(regSpec, bestWvlt='db8', smoothed_array=True, plot_fig=False)
    for s in range(0, nSpecs):
        # reg_norm[s, :] = normalize_spectrum(regSpec[s, :], normalize='tic')
        # reg_norm_ = _smooth_spectrum(regSpec[s, :], method='savgol', window_length=5, polyorder=2)
        # reg_norm[s, :] = normalize_spectrum(reg_norm_, normalize='tic')
        reg_norm[s, :] = normalize_spectrum(reg_smooth_[s, :], normalize='tic')
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
    # savecsv = os.path.join(regDir, '{}_reg_{}_{}.csv'.format(filename, regID, exprun))
    # df_pca_umap_hdbscan.to_csv(savecsv, index=False, sep=',')

    # +--------------------------------------+
    # |   Segmentation on PCA+UMAP+HDBSCAN   |
    # +--------------------------------------+
    # regInd = ImzObj.get_region_indices(regID)
    # xr, yr, zr, _ = ImzObj.get_region_range(regID)
    # xx, yy, _ = ImzObj.get_region_shape(regID)
    sarray = np.zeros([regArr.shape[0], regArr.shape[1]])
    for idx, coord in enumerate(regCoor):
        # print(idx, coord, ImzObj.coord2index.get(coord))
        xpos = coord[0]# - xr[0]
        ypos = coord[1]# - yr[0]
        sarray[xpos, ypos] = labels[idx] + 1    # to avoid making 0 as bg
    sarray = nnPixelCorrect(sarray, 20, 3)  # noisy pixel is labeled as 19 by hdbscan
    fig, ax = plt.subplots(figsize=(6, 8))
    sarrayIm = ax.imshow(sarray)
    fig.colorbar(sarrayIm)
    ax.set_title('reg{}: HDBSCAN labeling'.format(regID), fontsize=15, loc='center')
    plt.show()
    if save_rseg:
        namepy = os.path.join(regDir, 'reg_{}_{}_hdbscan-label.npy'.format(regID, exprun))
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
            labels = gmm.fit_predict(data_umap)
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
    savecsv = os.path.join(regDir, 'reg_{}_{}.csv'.format(regID, exprun))     # todo
    df_pca_umap_hdbscan_gmm.to_csv(savecsv, index=False, sep=',')

    nGs = retrace_columns(df_pca_umap_hdbscan_gmm.columns.values, 'gmm')
    df_gmm_labels = df_pca_umap_hdbscan_gmm.iloc[:, -nGs:]
    # print("gmm label: ", nGs)

    for (columnName, columnData) in df_gmm_labels.iteritems():
        print('Column Name : ', columnName)
        print('Column Contents : ', columnData.values)
        # regInd = ImzObj.get_region_indices(regID)
        # xr, yr, zr, _ = ImzObj.get_region_range(regID)
        # xx, yy, _ = ImzObj.get_region_shape(regID)
        sarray1 = np.zeros([regArr.shape[0], regArr.shape[1]])
        for idx, coord in enumerate(regCoor):
            xpos = coord[0] #- xr[0]
            ypos = coord[1] #- yr[0]
            sarray1[xpos, ypos] = columnData.values[idx] + 1
        fig, ax = plt.subplots(figsize=(6, 8))
        sarrayIm = ax.imshow(sarray1)
        fig.colorbar(sarrayIm)
        ax.set_title('reg{}: umap_{}'.format(regID, columnName), fontsize=15, loc='center')
        plt.show()
        if save_rseg:
            namepy = os.path.join(regDir, 'reg_{}_umap-{}.npy'.format(regID, columnName))
            np.save(namepy, sarray)
    return

def msmlfunc3(mspath, regID, threshold, exprun=None, downsamp_i=None, wSize=[2,2,1]):
    """
    downsampling ml enabled and imzl obj dependency minimized.
    mspath: path to .imzML file
    regID: region to be analysed
    threshold: how much variance needed for PCA
    exprun: name of the experimental run
    downsamp_i: index pixel to downsample with
    e.g.:
    """
    RandomState = 20210131
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
    plot_spec = True
    plot_pca = True
    plot_umap = True
    save_rseg = True
    # +------------------------------------+
    # |     read data and save region      |
    # +------------------------------------+
    dirname = os.path.dirname(mspath)
    basename = os.path.basename(mspath)
    filename, ext = os.path.splitext(basename)
    regDir = os.path.join(dirname, 'reg_{}'.format(regID))
    if not os.path.isdir(regDir):
        os.mkdir(regDir)
    regname = os.path.join(regDir, '{}_reg_{}.mat'.format(filename, regID))
    ImzObj = IMZMLExtract(mspath)
    if os.path.isfile(regname):
        matr = loadmat(regname)
        regArr = matr['array']
        regSpec = matr['spectra']
        regCoor = matr['coordinates']
    else:
        BinObj = Binning2(ImzObj, regID)
        regArr, regSpec, regCoor = BinObj.getBinMat()
        matr = {"spectra": regSpec, "array": regArr, "coordinates": regCoor, "info": "after peakpicking in Cardinal"}
        savemat(regname, matr)    # basename
    nSpecs, nBins = regSpec.shape
    print(">> There are {} spectrums and {} m/z bins".format(nSpecs, nBins))
    # +----------------+
    # |  downsampling  |
    # +----------------+
    if downsamp_i is not None:
        downArray, downSpec, downCoor = downSpatMS(regArr, downsamp_i, wSize)
        regSpec = copy.deepcopy(downSpec)
        regArr = copy.deepcopy(downArray)
        regCoor = copy.deepcopy(downCoor)
        nSpecs, nBins = regSpec.shape
        print(">> After downsampling {} spectrums and {} m/z bins".format(nSpecs, nBins))
        regDir = os.path.join(regDir, 'down_{}'.format(downsamp_i))
        if not os.path.isdir(regDir):
            os.mkdir(regDir)
    # +------------------------------+
    # |   normalize, standardize     |
    # +------------------------------+
    reg_norm = np.zeros_like(regSpec)
    _, reg_smooth_ = bestWvltForRegion(regSpec, bestWvlt='db8', smoothed_array=True, plot_fig=False)
    for s in range(0, nSpecs):
        reg_norm[s, :] = normalize_spectrum(reg_smooth_[s, :], normalize='tic')
        # reg_norm_ = _smooth_spectrum(regSpec[s, :], method='savgol', window_length=5, polyorder=2)
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
    # print(evr)
    evr_cumsum = np.cumsum(evr)
    # print(evr_cumsum)
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
    # savecsv = os.path.join(regDir, '{}_{}.csv'.format(filename, exprun))
    # df_pca_umap_hdbscan.to_csv(savecsv, index=False, sep=',')

    # +--------------------------------------+
    # |   Segmentation on PCA+UMAP+HDBSCAN   |
    # +--------------------------------------+
    # regInd = ImzObj.get_region_indices(regID)
    # xr, yr, zr, _ = ImzObj.get_region_range(regID)
    # xx, yy, _ = ImzObj.get_region_shape(regID)
    sarray = np.zeros([regArr.shape[0], regArr.shape[1]])
    for idx, coord in enumerate(regCoor):
        # print(idx, coord, ImzObj.coord2index.get(coord))
        xpos = coord[0]# - xr[0]
        ypos = coord[1]# - yr[0]
        sarray[xpos, ypos] = labels[idx] + 1    # to avoid making 0 as bg
    sarray = nnPixelCorrect(sarray, 20, 3)  # noisy pixel is labeled as 19 by hdbscan
    fig, ax = plt.subplots(figsize=(6, 8))
    sarrayIm = ax.imshow(sarray)
    fig.colorbar(sarrayIm)
    ax.set_title('reg{}: HDBSCAN labeling'.format(regID), fontsize=15, loc='center')
    plt.show()
    if save_rseg:
        namepy = os.path.join(regDir, '{}_hdbscan-label.npy'.format(exprun))
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
    savecsv = os.path.join(regDir, '{}.csv'.format(exprun))
    df_pca_umap_hdbscan_gmm.to_csv(savecsv, index=False, sep=',')

    nGs = retrace_columns(df_pca_umap_hdbscan_gmm.columns.values, 'gmm')
    df_gmm_labels = df_pca_umap_hdbscan_gmm.iloc[:, -nGs:]
    # print("gmm label: ", nGs)

    for (columnName, columnData) in df_gmm_labels.iteritems():
        print('Column Name : ', columnName)
        print('Column Contents : ', columnData.values)
        # regInd = ImzObj.get_region_indices(regID)
        # xr, yr, zr, _ = ImzObj.get_region_range(regID)
        # xx, yy, _ = ImzObj.get_region_shape(regID)
        sarray1 = np.zeros([regArr.shape[0], regArr.shape[1]])
        for idx, coord in enumerate(regCoor):
            xpos = coord[0]# - xr[0]
            ypos = coord[1]# - yr[0]
            sarray1[xpos, ypos] = columnData.values[idx] + 1
        fig, ax = plt.subplots(figsize=(6, 8))
        sarrayIm = ax.imshow(sarray1)
        fig.colorbar(sarrayIm)
        ax.set_title('reg{}: umap_{}'.format(regID, columnName), fontsize=15, loc='center')
        plt.show()
        if save_rseg:
            namepy = os.path.join(regDir, 'umap-{}_{}.npy'.format(exprun, columnName))
            np.save(namepy, sarray)
    return

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet=None, level=2):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    num_ = 1.0
    den_ = 0.6745*3
    sigma = (num_/den_) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def cosineSimTrial(imzObj, spn1, spn2):
    """
    The cosine similarity used in matchms is equal to the one in scipy
    check cosine similarity between matchms and scipy
    Arg:
        imzObj: imzml data obj
        sp1: spectrum number 1
        sp2: spectrum number 2
    Return:
        similarity score
    e.g.: cosineSimTrial(imze, 19264, 17451)
    """
    spectrum1 = Spectrum(mz=np.array(imzObj.parser.getspectrum(spn1)[0]).astype(float),
                         intensities=np.array(imzObj.parser.getspectrum(spn1)[1]).astype(float),
                         metadata={'id': spn1})
    spectrum2 = Spectrum(mz=np.array(imzObj.parser.getspectrum(spn2)[0]).astype(float),
                         intensities=np.array(imzObj.parser.getspectrum(spn2)[1]).astype(float),
                         metadata={'id': spn2})
    # spectrum1_filtered = normalize_intensities(spectrum1)
    score_ = calculate_scores([spectrum1],
                              [spectrum2],
                              CosineGreedy())
    # score_ = CosineGreedy.pair(spectrum1, spectrum2)
    mms_score = score_.scores[0][0].item()[0]
    scp_score_mms = 1 - cosine(spectrum1.peaks.intensities,
                           spectrum2.peaks.intensities)
    #updated: to check consistency between mms and imz intensity.
    scp_score_imz = 1 - cosine(imzObj.parser.getspectrum(spn1)[1],
                           imzObj.parser.getspectrum(spn2)[1])

    score_ = calculate_scores([spectrum1],
                              [spectrum2],
                              CosineHungarian())
    print("mms: {}, scp_mms: {}, scp_imz: {}".format(mms_score,
                                                 scp_score_mms,
                                                 scp_score_imz))
    print("Hungarian cosine: {}".format(score_.scores[0][0].item()[0]))

    fig, ax = plt.subplots(dpi=100)
    ax.plot(spectrum1.peaks.intensities, color=(0.9, 0, 0), linewidth=1.5, label='spn1')
    ax.set_xlabel("m/z", fontsize=12)
    ax.set_ylabel("intensity", fontsize=12, color=(0.9, 0, 0))
    ax.legend(loc='upper center')
    ax.grid()
    ax.plot(spectrum2.peaks.intensities, color=(0, 0, 0.9), linewidth=1.5, label='spn2', alpha=0.5)
    ax.set_xlabel("m/z", fontsize=12)
    ax.set_ylabel("intensity", fontsize=12, color=(0, 0, 0.9), alpha=0.5)
    ax.legend(loc='upper right')
    fig.suptitle('Spectrums {}-{}'.format(spn1, spn2), fontsize=12, y=1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout(pad=0.2)
    plt.show()
    return None

def zeroBaseSpec(specNZ):
    spec_z = copy.deepcopy(specNZ)
    spec_z[specNZ > 0] = 0
    spec_z = specNZ - spec_z
    return spec_z

def waveletPickwSim(signal, plot_fig=True, verbose=True):
    """
    pick which wavelet to smooth the spectra with best similarity metric
    e.g.: _, wv, sm = waveletPickwSim(signal, plot_fig=False, verbose=False)
    """
    wvltList = pywt.wavelist()
    discreteWvList = []
    continuousWvList = []
    for w in wvltList:
        try:
            w1 = pywt.Wavelet(w)
            discreteWvList.append(w)
        except:
            # print(w, "is not discrete")
            continuousWvList.append(w)
            pass
    _similarity = -np.inf
    _similarityList = []
    _wv = 0
    _wvList = []
    _idx = 0
    _filtered = np.zeros_like(signal)
    for idx, wv in enumerate(discreteWvList):
        filtered = wavelet_denoising(signal, wavelet=wv)
        assert (signal.shape == filtered.shape)
        filtered = zeroBaseSpec(filtered)
        if (1 - cosine(signal, filtered)) > _similarity:
            _similarity = 1 - cosine(signal, filtered)
            _wv = wv
            _idx = idx
            _filtered = filtered
    # print("idx: {} #wv: {}".format(idx, len(discreteWvList)))
    if verbose:
        print("Similarity: {} found mostly with #{}{} wavelet".format(_similarity, _idx, _wv))
    if plot_fig:
        fig, ax = plt.subplots(dpi=100)
        ax.plot(signal, color=(0.9, 0, 0), linewidth=1.5, label='raw')
        ax.set_xlabel("m/z(shifted)", fontsize=12)
        ax.set_ylabel("intensity", fontsize=12)
        ax.grid()
        ax.plot(_filtered, color=(0, 0, 0.9), linewidth=1.5, label='{} filtered'.format(_wv), alpha=0.5)
        ax.legend(loc='upper right')
        fig.suptitle('wavelet smoothing', fontsize=12, y=1, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.tight_layout(pad=0.2)
        plt.show()
    return _filtered, _wv, _similarity

def bestWvltForRegion(spectra_array, bestWvlt, smoothed_array=True, plot_fig=True):
    """
    spectra array is of shape: (#Spec, #m/z), where each row value is intensity.
    >> (4587, 1332)
    bestwvlt: e.g.: 'db8' ; if not given finds.
    smooth_array: if True returns smoothed spectra array with best wavelet.
    plot_fig: plots figure if True.
    e.g.: bestWvltForRegion(spec_data1, bestWvlt='db8', smoothed_array=True, plot_fig=True)
    """
    if bestWvlt is None:
        wvList = []
        smList = []
        for nS in tqdm(range(spectra_array.shape[0]), desc = '#spectra%'):
            signal = spectra_array[nS, :]   #   copy.deepcopy(spectra_array[nS, :])
            _, wv, sm = waveletPickwSim(signal, plot_fig=False, verbose=False)
            wvList.append(wv)
            smList.append(sm)
        if plot_fig:
            keys, counts = np.unique(wvList, return_counts=True)
            plt.bar(keys, counts)
            plt.xticks(rotation='vertical')
            plt.show()
        bestWvlt = max(set(wvList), key=wvList.count)
        print("Best performing wavelet: {}".format(bestWvlt))

    if plot_fig:
        nS = np.random.randint(spectra_array.shape[0])
        signal = copy.deepcopy(spectra_array[nS, :])
        filtered = wavelet_denoising(signal, wavelet=bestWvlt)
        assert (signal.shape == filtered.shape)
        # filtered_ = copy.deepcopy(filtered)
        # filtered_[filtered > 0] = 0  # for baseline correction to 0
        # filtered = filtered - filtered_
        filtered = zeroBaseSpec(filtered)
        fig, ax = plt.subplots(dpi=100)
        ax.plot(signal, color=(0.9, 0, 0), linewidth=1.5, label='raw')
        ax.set_xlabel("m/z(shifted)", fontsize=12)
        ax.set_ylabel("intensity", fontsize=12)
        ax.grid()
        ax.plot(filtered, color=(0, 0, 0.9), linewidth=1.5, label='{} filtered'.format(bestWvlt), alpha=0.5)
        ax.legend(loc='upper right')
        fig.suptitle('wavelet smoothing', fontsize=12, y=1, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.tight_layout(pad=0.2)
        plt.show()

    if smoothed_array:
        smoothed_array_ = np.zeros_like(spectra_array)
        for nS in range(spectra_array.shape[0]):
            signal = copy.deepcopy(spectra_array[nS, :])
            filtered = wavelet_denoising(signal, wavelet=bestWvlt)
            assert (signal.shape == filtered.shape)
            filtered = zeroBaseSpec(filtered)
            smoothed_array_[nS, :] = filtered
        return bestWvlt, smoothed_array_
    else:
        return bestWvlt

def downSpatMS(msArray, i_, wSize):
    """
    spatially downsample an MS image array(region)
    e.g.: downArray, downSpec, downCoor = downSpatMS(spectra_array, 0, [2, 2, 1])
    """
    if i_ >= np.prod(wSize):
        raise ValueError("The index of window must be within")
    pad_width = []
    for i in range(len(wSize)):
        if wSize[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if msArray.shape[i] % wSize[i] != 0:
            after_width = wSize[i] - (msArray.shape[i] % wSize[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))
    cval = 0
    msArray = np.pad(msArray, pad_width=pad_width, mode='constant',
                     constant_values=cval)  # pads the image if not divisible.
    # print(msArray.shape)
    # if len(wSize) == 3:
    #     r_, c_, d_ = np.unravel_index(i_, (wSize))
    #     msArray_down = np.zeros([int(msArray.shape[0] / wSize[0]),
    #                              int(msArray.shape[1] / wSize[1]),
    #                              int(msArray.shape[2] / wSize[2])])
    #     for r in range(0, msArray.shape[0], wSize[0]):
    #         for c in range(0, msArray.shape[1], wSize[1]):
    #             for d in range(0, msArray.shape[2], wSize[2]):
    #                 msArray_down[int(r / wSize[0]), int(c / wSize[1]), int(d / wSize[2])] = msArray[r + r_, c + c_, d + d_]
    # else:
    r_, c_, d_ = np.unravel_index(i_, (wSize))
    msArray_down = np.zeros([int(msArray.shape[0] / wSize[0]),
                             int(msArray.shape[1] / wSize[1]),
                             msArray.shape[2]])
    for r in range(0, msArray.shape[0], wSize[0]):
        for c in range(0, msArray.shape[1], wSize[1]):
            msArray_down[int(r / wSize[0]), int(c / wSize[1]), :] = msArray[r + r_, c + c_, :]
    image_sum = np.sum(msArray_down, axis=2)
    image_bw = copy.deepcopy(image_sum)
    image_bw[image_bw > 0] = 1
    spectra_down = np.zeros([int(np.sum(image_bw)), msArray.shape[-1]], dtype=msArray.dtype)
    nSd = 0
    coordList_down = []
    for r in range(0, msArray_down.shape[0]):
        for c in range(0, msArray_down.shape[1]):
            if image_bw[r, c] == 1:
                spectra_down[nSd, :] = msArray_down[r, c, :]
                coordList_down.append((r, c))
                nSd += 1
    # print(spectra_down.shape, len(coordList_down))
    # print(coordList_down)
    return msArray_down, spectra_down, coordList_down

def matchSpecLabel(seg1, seg2, arr1, arr2, plot_fig=True): #TODO: implement for multi seg
    """
    seg1 & seg2: segmentation file path(.npy)
    Comparison of segmentation between two region arrays
    """
    specDict = {}
    label1 = np.load(seg1)
    label2 = np.load(seg2)

    for l in range(1, len(np.unique(label1))):  # to avoid 0-background
        label1_ = copy.deepcopy(label1)
        label1_[label1 != l] = 0
        spec = np.mean(arr1[np.where(label1_)], axis=0)
        spec = {"{}_{}".format(1, l): spec}
        specDict.update(spec)

    for l in range(1, len(np.unique(label2))):  # to avoid 0-background
        label2_ = copy.deepcopy(label2)
        label2_[label2 != l] = 0
        spec = np.mean(arr2[np.where(label2_)], axis=0)
        spec = {"{}_{}".format(2, l): spec}
        specDict.update(spec)

    def cosineSim(spectrum1, spectrum2):
        return 1 - cosine(spectrum1,
                          spectrum2)
    spec_df = pd.DataFrame(specDict)
    method_ = ['pearson', 'spearman', 'kendall', cosineSim]
    corr = spec_df.corr(method=method_[3])
    # if corr.
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(corr, ax=ax, annot=True,
                     vmin=0, vmax=1, center=0.5,
                     cbar_ax=cbar_ax,
                     cbar_kws={"orientation": "horizontal"},
                     cmap='Greys', #"YlGnBu",
                     linewidths=0.5,
                     square=True)
    plt.show()
    if plot_fig:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        arrays = [label1, label2]
        title = ['Seg#1', 'Seg#2']
        fig, axs = plt.subplots(1, 2, figsize=(10, 8), dpi=200, sharex=False)
        for ar, tl, ax in zip(arrays, title, axs.ravel()):
            im = ax.imshow(ar) #, cmap='twilight') #cm)
            ax.set_title(tl, fontsize=20)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, ax=ax)
        plt.show()
    return specDict

def matchSpecLabel_(plot_fig, *segs, **kwarr): #TODO: implement for multi seg
    """
    seg1 & seg2: segmentation file path(.npy)
    Comparison of segmentation between two region arrays
    """
    specDict = {}
    segList = []
    title = []
    for i, (s, (key, value)) in enumerate(zip(segs, kwarr.items())):
        label = np.load(s)
        segList.append(label)
        title.append(key)
        for l in range(1, len(np.unique(label))):
            label_ = copy.deepcopy(label)
            label_[label != l] = 0
            spec = np.mean(value[np.where(label_)], axis=0)
            spec = {"{}_{}".format(i+1, l): spec}
            specDict.update(spec)

    def cosineSim(spectrum1, spectrum2):
        return 1 - cosine(spectrum1,
                          spectrum2)
    spec_df = pd.DataFrame(specDict)
    method_ = ['pearson', 'spearman', 'kendall', cosineSim]
    corr = spec_df.corr(method=method_[3])
    # if corr.
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(corr, ax=ax, annot=True,
                     vmin=0, vmax=1, center=0.5,
                     cbar_ax=cbar_ax,
                     cbar_kws={"orientation": "horizontal"},
                     cmap='Greys', #"YlGnBu",
                     linewidths=0.5,
                     # xticklabels='vertical',
                     # yticklabels='horizontal',
                     square=False)
    ax.set_yticklabels(specDict.keys(), rotation=0)
    ax.set_xticklabels(specDict.keys(), rotation=90)
    ax.xaxis.tick_top()
    # plt.show()
    if plot_fig:
        fig, axs = plt.subplots(1, i+1, figsize=(12, 16), dpi=300, sharex=False)
        for ar, tl, ax in zip(segList, title, axs.ravel()):
            im = ax.imshow(ar) #, cmap='twilight') #cm)
            ax.set_title(tl, fontsize=20)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, ax=ax)
        plt.show()
    return specDict