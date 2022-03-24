import os
from glob import glob
import numpy as np
from imzml import IMZMLExtract, Binning2, normSpec, normalize_spectrum
from scipy.io import savemat, loadmat
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
import hdbscan
from sklearn.mixture import GaussianMixture as GMM
import matplotlib as mtl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pywt
import pandas as pd
import time
import copy

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
def generate_nComponentList(n_class, span):
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

# +----------------+
# |  plot spectra  |
# +----------------+
if __name__ != '__main__':
    nS = 210
    fig, ax = plt.subplots(4, 1, figsize=(20, 8), dpi=200)
    ax[0].plot(data[nS, :])
    plt.title("raw spectrum")
    ax[1].plot(data_norm[nS, :])
    plt.title("'tic' norm")
    ax[2].plot(data_norm_ss[nS, :])
    plt.title("standardized")
    ax[3].plot(np.mean(data, axis=0))
    plt.title("mean spectra(region 1)")
    plt.suptitle("Processing comparison of Spec #{}".format(nS))
    plt.show()

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

    signal = data[nS, :]

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

# +------------+
# |    PCA     |
# +------------+
if __name__ != '__main__':
    pca_all = PCA(random_state=RandomState)
    pcs_all = pca_all.fit_transform(pixel_feature)
    # pcs_all=pca_all.fit_transform(oldLipid_mm_norm)
    pca_range = np.arange(1, pca_all.n_components_, 1)
    print("PCA: number of components #{}".format(pca_all.n_components_))
    printStat(pcs_all)
    evr = pca_all.explained_variance_ratio_
    print(evr)
    evr_cumsum = np.cumsum(evr)
    print(evr_cumsum)
    threshold = 0.95  # to choose PCA variance
    cut_evr = find_nearest(evr_cumsum, threshold)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print("Nearest variance to threshold {:.4f} explained by PCA components {}".format(cut_evr, nPCs))
    df_pixel_rep = pd.DataFrame(data=pcs_all[:, 0:nPCs], columns=['PC_%d' % (i+1) for i in range(nPCs)])
    # df_pixel_rep.insert(0, 'spectrum_index', [i for i in range(nSpecs)])
    # df_pixel_rep.insert(0, 'label_index', [0 for i in range(nSpecs)]) # this is probably not required?
    print(df_pixel_rep, df_pixel_rep.shape)
    print(pca_all.n_features_)
    if __name__ == '__main__':
        # matr = {"data": df_pixel_rep.values, "info": "data after pca,variance:{}, components:{}".format(threshold, nPCs)}
        # savemat(os.path.join(oldCDir, 'data_pca.mat'), matr)
        savecsv = os.path.join(oldCDir, '{}.csv'.format(exprun))
        df_pixel_rep.to_csv(savecsv, index=False, sep=',')
        # df_pixel_rep.to_csv(savecsv, index=True, index_label='Spectrum_index', sep=',')

# +------------------+
# |    plot PCA      |
# +------------------+
if __name__ != '__main__':
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
    # plt.grid("On")
    plt.show()

if __name__ != '__main__':
    plt.figure(figsize=(12, 10), dpi=200)
    plt.scatter(df_pixel_rep.PC_1, df_pixel_rep.PC_2, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
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
if __name__ != '__main__':
    # pca_path = os.path.join(oldCDir, '{}.csv'.format(exprun))
    # pca_path = os.path.join(oldCDir, 'oldLipid1.mat')
    # df_pixel_rep = pd.read_csv(pca_path)
    # df_pixel_rep = loadmat(pca_path)
    # df_pixel_rep = df_pixel_rep['data']
    reducer = UMAP(n_neighbors=12,  # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=3,  # default 2, The dimension of the space to embed into.
               metric='cosine',  # default 'euclidean', The metric to use to compute distances in high dimensional space.
               n_epochs=1000,  # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
               learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral',  # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.025,  # default 0.1, The effective minimum distance between embedded points.
               spread=1.0,  # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
               repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=RandomState, # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
               #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=True, # default False, Controls verbosity of logging.
               unique=False # default False, Controls if the rows of your data should be uniqued before being embedded.
              )
    data_umap = reducer.fit_transform(df_pixel_rep.values) # on pca
    for i in range(reducer.n_components):
        df_pixel_rep.insert(df_pixel_rep.shape[1], column='umap_{}'.format(i + 1), value=data_umap[:, i])

    if __name__ != '__main__':
        savecsv = os.path.join(oldCDir, '{}.csv'.format(exprun))
        df_pixel_rep.to_csv(savecsv, index=False, sep=',')
        matr = {"data": df_pixel_rep.values, "n_neighbors": reducer.n_neighbors, "info": "norm/pca/umap: {}".format(reducer.get_params())}
        savemat(os.path.join(oldCDir, '{}.mat'.format(exprun)), matr)

if __name__ != '__main__':
    data_pca_umap_path = os.path.join(oldCDir, 'all2022-03-21-19-02-17.csv')
    data_pca_umap = pd.read_csv(data_pca_umap_path)
    print(data_pca_umap.columns)
    print(data_pca_umap.values[:, 0])
    # data_umap_obj = loadmat(data_umap_path)
    # print(sorted(data_umap_obj.keys()))
    # data_umap = data_umap_obj['data']
    data_umap = data_pca_umap.values[:, 9:]

# +---------------+
# |   UMAP plot   |
# +---------------+
if __name__ != '__main__':
    plt.figure(figsize=(12, 10))
    plt.scatter(data_umap[:, 0], data_umap[:, 1], facecolors='None', edgecolors=cm.tab20(10), alpha=0.5)
    plt.xlabel('UMAP1', fontsize=30) #only difference part from last one
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
    # SaveDir = OutputFolder + '\\exploratory results\\UMAP_plot.png'
    # plt.savefig(SaveDir)
    # plt.close()

    # import umap.plot
    #
    # umap.plot.points(data_umap)

# +-------------+
# |   HDBSCAN   |
# +-------------+
if __name__ != '__main__':
    HDBSCAN_soft = False
    min_cluster_size = 250
    min_samples = 30
    cluster_selection_method = 'eom' #eom
    if HDBSCAN_soft:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_method=cluster_selection_method, prediction_data=True)\
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
    # SaveDir = OutputFolder + '\\exploratory results\\UMAP_HDBSCAN_plot.png'
    # plt.savefig(SaveDir)
    # plt.close()
    # print(clusterer.__dict__)
    chart(data_umap, labels)
    plt.suptitle("n_neighbors={}, Cosine metric".format(12))

    if __name__ != '__main__':
        data_pca_umap.insert(data_pca_umap.shape[1], column='hdbscan_labels', value=labels)
        savecsv = os.path.join(oldCDir, '{}.csv'.format(exprun))
        data_pca_umap.to_csv(savecsv, index=False, sep=',')
        # matr = {"data": labels, "info": 'after umap'}
        # savemat(os.path.join(oldCDir, 'label_umap_hdbscan_{}.mat'.format(exprun)), matr)

# labelpath = glob(os.path.join(oldCDir, 'label_umap_hdbscan.mat'))
# # print(labelpath[0])
# labels_obj = loadmat(labelpath[0])
# # print(sorted(data.keys()))
# labels = labels_obj['data'].squeeze()
# print(labels)


if __name__ != '__main__':
    oldCFile = glob(os.path.join(oldCDir, '*.imzML'))
    print(oldCFile[0])
    oldCImz = IMZMLExtract(oldCFile[0])
    ##
    regionID = 1
    # labelpath = glob(os.path.join(oldCDir, 'label_umap_hdbscan.mat'))
    # # print(labelpath[0])
    # labels_obj = loadmat(labelpath[0])
    # # print(sorted(data.keys()))
    # labels = labels_obj['data'].squeeze()
    pca_path = os.path.join(oldCDir, 'hdbscan_2022-03-22-12-41-33.csv')
    df_pixel_rep_label = pd.read_csv(pca_path)
    print(df_pixel_rep_label.columns)
    labels = df_pixel_rep_label['hdbscan_labels'].values
    print("labels: ", labels) #.squeeze()) #np.array(labels, dtype=object))
    print(["{}:{}".format(l, np.sum(labels == l)) for l in np.unique(labels)])
    if __name__ == '__main__':
        regInd = oldCImz.get_region_indices(regionID)
        xr, yr, zr, _ = oldCImz.get_region_range(regionID)
        xx, yy, _ = oldCImz.get_region_shape(regionID)
        sarray1 = np.zeros([xx, yy])
        # for idx,coord in enumerate(oldLipid1[2]):
        #     print(idx, coord, assignment[idx])
        labels[np.where(labels == 19)] = 0
        for idx, coord in enumerate(regInd):
            print(idx, coord, oldCImz.coord2index.get(coord))
            xpos = coord[0] - xr[0]
            ypos = coord[1] - yr[0]
            sarray1[xpos, ypos] = labels[idx] + 1
        fig, ax = plt.subplots(figsize=(6, 8))
        sarrayIm = ax.imshow(sarray1)
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(sarrayIm)
        ax.set_title('reg1: HDBSCAN labeling', fontsize=15, loc='center')
        plt.show()

# +---------------+
# |   tree plots  |
# +---------------+
if __name__ != '__main__':
    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    plt.show()
    clusterer.condensed_tree_.plot()
    plt.show()
    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    plt.show()

# +-----------------+
# |       GMM       |
# +-----------------+
if __name__ != '__main__':
    pca_path = os.path.join(oldCDir, 'hdbscan_2022-03-22-12-41-33.csv')
    df_pixel_rep_label = pd.read_csv(pca_path)
    print(df_pixel_rep_label.columns[9:12])
    # df_pixel_rep['label_index'] = labels
    #
    # print(df_pixel_rep_label.values[:, 9:])
    #
    # savecsv = os.path.join(oldCDir, 'df_pixel_rep_label.csv')
    # df_pixel_rep.to_csv(savecsv, index=False, sep=',')

if __name__ != '__main__':
    n_components = 5
    span = 5
    n_component = generate_nComponentList(n_components, span)
    print("n_component: ", n_component)
    nPCs = 9    #  retrace_columns(df_pixel_rep.columns.values, 'PC')   # 9
    print("nPCs: ", nPCs)
    pixel_rep = df_pixel_rep_label.values.astype(np.float64)
    pcs_umap = pixel_rep[:, 9:12] #[:, 0:12]
    print(pcs_umap.shape)
    repeat = 2  # integer
    df_pixel_label = pd.DataFrame(df_pixel_rep_label['hdbscan_labels']) #pd.DataFrame(data=df_pixel_rep_label[['label_index', 'spectrum_index']].values.astype(int), columns=['label_index', 'spectrum_index'])
    print(df_pixel_label)

if __name__ != '__main__':
    for i in range(repeat):  # may repeat several times
        for j in range(n_component.shape[0]):  # ensemble with different n_component value
            StaTime = time.time()
            gmm = GMM(n_components=n_component[j], max_iter=5000)  # max_iter does matter, no random seed assigned
            labels = gmm.fit_predict(pcs_umap)
            # save data
            index = j + 1 + i * n_component.shape[0]
            title = 'gmm' + str(index) + '_' + str(n_component[j]) + '_' + str(i)
            df_pixel_label[title] = labels

            SpenTime = (time.time() - StaTime)

            # progressbar
            print('{}/{}, finish classifying {}, running time is: {} s'.format(index, repeat * span, title,
                                                                           round(SpenTime, 2)))
    print(df_pixel_label)
    oldCFile = glob(os.path.join(oldCDir, '*.imzML'))
    oldCImz = IMZMLExtract(oldCFile[0])
    regionID = 1

    for (columnName, columnData) in df_pixel_label.iteritems():
        print('Column Name : ', columnName)
        print('Column Contents : ', columnData.values)

        regInd = oldCImz.get_region_indices(regionID)
        xr, yr, zr, _ = oldCImz.get_region_range(regionID)
        xx, yy, _ = oldCImz.get_region_shape(regionID)
        sarray1 = np.zeros([xx, yy])
        # for idx,coord in enumerate(oldLipid1[2]):
        #     print(idx, coord, assignment[idx])
        labels[np.where(labels == 19)] = 0
        for idx, coord in enumerate(regInd):
            # print(idx, coord, oldCImz.coord2index.get(coord))
            xpos = coord[0] - xr[0]
            ypos = coord[1] - yr[0]
            sarray1[xpos, ypos] = columnData.values[idx] + 1
        fig, ax = plt.subplots(figsize=(6, 8))
        sarrayIm = ax.imshow(sarray1)
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(sarrayIm)
        ax.set_title('reg 1: UMAP w GMM {} labelling'.format(columnName), fontsize=15, loc='center')
        plt.show()

    frames = [df_pixel_rep_label, df_pixel_label]
    pca_umap_hdbscan_gmm = pd.concat(frames)
    savecsv = os.path.join(oldCDir, '{}.csv'.format(exprun))
    pca_umap_hdbscan_gmm.to_csv(savecsv, index=False, sep=',')

mspath = glob(os.path.join(oldCDir, '*.imzML'))[0]
# print(mspath)
dirname = os.path.dirname(mspath)
basename = os.path.basename(mspath)
filename, ext = os.path.splitext(basename)
regID = 3
# print(os.path.join(dirname, str(filename + '_reg_{}.mat'.format(regID))))

ImzObj = IMZMLExtract(mspath)
regInd = ImzObj.get_region_indices(regID)
print(regInd)
xr, yr, zr, _ = ImzObj.get_region_range(regID)
print(xr, yr, zr, _)    # (704, 798) (146, 212) (1, 1) 1332
xx, yy, _ = ImzObj.get_region_shape(regID)
print(xx, yy, _)    # 95 67 1332

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
        reg_norm[s, :] = normalize_spectrum(regSpec[s, :], normalize='tic')
        # printStat(data_norm)
    reg_norm_ss = makeSS(reg_norm).astype(np.float64)

    # +----------------+
    # |  plot spectra  |
    # +----------------+
    if plot_spec:
        nS = np.random.randint(nSpecs)
        fig, ax = plt.subplots(4, 1, figsize=(20, 8), dpi=200)
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

    data = reg_norm_ss
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
    savecsv = os.path.join(dirname, '{}_reg_{}_{}.csv'.format(filename, regID, exprun))
    df_pca_umap_hdbscan.to_csv(savecsv, index=False, sep=',')

    # +--------------------------------------+
    # |   Segmentation on PCA+UMAP+HDBSCAN   |
    # +--------------------------------------+
    regInd = ImzObj.get_region_indices(regID)
    xr, yr, zr, _ = ImzObj.get_region_range(regID)
    xx, yy, _ = ImzObj.get_region_shape(regID)
    sarray = np.zeros([xx, yy])
    # labels[np.where(labels == 19)] = 0
    for idx, coord in enumerate(regInd):
        print(idx, coord, ImzObj.coord2index.get(coord))
        xpos = coord[0] - xr[0]
        ypos = coord[1] - yr[0]
        sarray[xpos, ypos] = labels[idx] + 1
    fig, ax = plt.subplots(figsize=(6, 8))
    sarrayIm = ax.imshow(sarray)
    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    fig.colorbar(sarrayIm)
    ax.set_title('reg{}: HDBSCAN labeling'.format(regID), fontsize=15, loc='center')
    plt.show()
    namepy = os.path.join(dirname, '{}_reg_{}_{}.npy'.format(filename, regID, exprun))
    np.save(namepy, sarray)
    return