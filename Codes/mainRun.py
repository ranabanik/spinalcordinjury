import os
import copy
from glob import glob
import math
import numpy as np
import pywt
from Codes.Utilities import msmlfunc5, matchSpecLabel2, ImzmlAll, rawVSprocessed
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib as mtl
# mtl.use('TkAgg')    # required for widget slider...
from scipy.io import loadmat, savemat
from scipy.stats import stats
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import time
from Codes.imzml import IMZMLExtract, normalize_spectrum, getionimage
from pyimzml.ImzMLParser import _bisect_spectrum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from pyimzml.ImzMLParser import ImzMLParser
from ms_peak_picker import pick_peaks
import h5py

# posLip = r'C:\Data\210427-Chen_poslip' #r'C:\Data\PosLip'
posLip = r'C:\Data\210427-Chen_poslip' #'/media/banikr/DATA/MALDI/demo_banikr_'
posLipNew = r'C:\Data\220211_reyzerml_IMC_380_plate4A_poslipids'  #'/media/banikr/DATA/MALDI/220210_reyzerml_IMC_380_plate1A_poslipids-NEW'
posLipNew2 = r'/media/banikr/DATA/MALDI/220210_reyzerml_IMC_380_plate2A_poslipid-NEW'
posLipNew3 = r'/media/banikr/DATA/MALDI/220211_reyzerml_IMC_380_plate3A_poslipids'
posLipNew4 = r'/media/banikr/DATA/MALDI/220211_reyzerml_IMC_380_plate4A_poslipids'

pathList = [posLip, posLipNew] #, posLipNew2, posLipNew3, posLipNew4]
mspathList = [glob(os.path.join(mp, '*.imzML'))[0] for mp in pathList]
# print(mspathList)

# ImzObj = ImzmlAll(mspathList[0])
msmlfunc5(mspathList[0], regID=1, threshold=0.99, exprun='pca_new')

if __name__ != '__main__':
    # for s in range(nPixels):
    #     ticnorm = tictic(spectra[nS, :]) #normalize_spectrum(spectra[nS, :], normalize='tic')
    #     reg_norm[s, :] = (ticnorm-min(ticnorm))/(max(ticnorm)-min(ticnorm))
    #     if s == nS:
    #         plt.hist(ticnorm)
    #         plt.show()
    #         plt.hist(reg_norm[s, :])
    #         plt.show()
    #         fig, ax = plt.subplots(dpi=100)
    #         ax.plot(peakmzs, ticnorm, 'r', alpha=1.0)
    #         ax0 = ax.twinx()
    #         ax0.plot(peakmzs, reg_norm[s, :], 'b', alpha=0.5)
    #         plt.show()
    #
    # print(max(reg_norm.ravel()), min(reg_norm.ravel()))
    mz_feature = images_flat.T
    # from sklearn.preprocessing import StandardScaler as SS
    #
    # from Utilities import makeSS
    # pixel_feature_std = makeSS(reg_norm) #SS().fit_transform(reg_norm)
    # nPCs = 13
    # pca = PCA(random_state=20210131) #, n_components=nPCs)
    # pcs = pca.fit_transform(pixel_feature_std)
    loadings = pca.components_.T
    # sum of squared loadings
    SSL = np.sum(loadings**2, axis=0)
    HC_method = 'ward'
    HC_metric = 'euclidean'
    import scipy.cluster.hierarchy as sch
    import matplotlib.cm as cm
    Y = sch.linkage(images_flat, method=HC_method, metric=HC_metric)
    Z = sch.dendrogram(Y, no_plot=True)
    HC_idx = Z['leaves']
    HC_idx = np.array(HC_idx)
    thr_dist = 78
    # plot it
    plt.figure(figsize=(15, 10))
    Z = sch.dendrogram(Y, color_threshold=thr_dist)
    plt.title('hierarchical clustering of ion images \n method: {}, metric: {}, threshold: {}'.format(
        HC_method, HC_metric, thr_dist))
    plt.show()
    # SaveDir = OutputFolder + '\\HC_dendrogram.png'
    # plt.savefig(SaveDir, dpi=dpi)
    # plt.close()

    ## 2. sort features with clustering results
    mz_feature_sorted = mz_feature[HC_idx]

    def _2d_to_3d(array2d, Coord, regionshape):
        nPixels, nMz = array2d.shape
        array3d = np.zeros([regionshape[0], regionshape[1], nMz])
        for idx, c in enumerate(Coord):
            array3d[c[0], c[1], :] = array2d[idx, :]
        return array3d

    images = _2d_to_3d(spectra, localCoor, regionshape)
    print("images.shape", images.shape)
    # plot it
    fig = plt.figure(figsize=(10, 10))
    axmatrix = fig.add_axes([0.10, 0, 0.80, 0.80])
    im = axmatrix.matshow(mz_feature_sorted, aspect='auto', origin='lower', cmap=cm.YlGnBu, interpolation='none')
    fig.gca().invert_yaxis()
    plt.show()
    # colorbar
    axcolor = fig.add_axes([0.96, 0, 0.02, 0.80])
    cbar = plt.colorbar(im, cax=axcolor)
    axcolor.tick_params(labelsize=10)

    ## 3. organize clusters with SSL
    # 1. organize ion images according to labels 2. generate average ion images
    HC_labels = sch.fcluster(Y, thr_dist, criterion='distance')

    # prepare label data
    elements, counts = np.unique(HC_labels, return_counts=True)
    print(elements, '\n', counts, ">>")
    # prepare imgs_std data
    # imgs_std = pixel_feature_std.T.reshape(120, NumLine, NumSpePerLine)
    # prepare accumulators
    mean_imgs = []
    total_SSLs = []

    for label in elements:
        idx = np.where(HC_labels == label)[0]

        # total SSL
        total_SSL = np.sum(SSL[idx])
        # imgs in the cluster
        # current_cluster = imgs_std[idx]
        # average img
        # mean_img = np.mean(imgs_std[idx], axis=0)

        # accumulate data
        total_SSLs.append(total_SSL)
        # mean_imgs.append(mean_img)

    print('Finish ion image clustering, next step: L1.2.3 sort clusters and plot')

def _boxplot(data, labels):
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)
    # labels = ['old', 'new']
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
    # labels = ['max spec', 'corrected']
    bplot = ax1.boxplot(list(data),
                        notch=False, sym='+', vert=True,
                        patch_artist=True, whis=1.5, labels=labels,
                        medianprops=medianprops, meanprops=meanlineprops,
                        showfliers=True, showmeans=True, meanline=True)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='gray',
                   alpha=0.6)
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='',
        xlabel='Data',
        ylabel='Value')
    ax1.set_title('Median of spectra', fontsize=16)
    # colors = ['maroon', 'darkblue', 'orangered', 'olive', 'sienna']
    # colors = ['maroon', 'darkblue', 'orangered', 'olive', 'sienna']
    medians = [bplot['medians'][i].get_ydata()[0] for i in range(len(labels))]
    pos = np.arange(len(labels)) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    for tick, label in zip(range(len(labels)), ax1.get_xticklabels()):
        ax1.text(pos[tick], .97, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', fontsize=10,
                 weight='bold',
                 color='firebrick')  # colors[tick])
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    for patch in bplot['boxes']:  # , colors):
        patch.set_facecolor('slategrey')
    fig.show()

# +--------------------------------------+
# |   resampling > peak-picking image    |
# +--------------------------------------+
if __name__ != '__main__':
    ImzObj = ImzmlAll(mspathList[0])
    regID = 1
    tol = 0.02
    regname = os.path.join(os.path.dirname(mspathList[0]), 'reg_{}_tol_{}.h5'.format(regID, tol))
    f = h5py.File(regname, 'r')
    # print(f.keys())
    r2d = np.array(f['2D'])
    r3d = np.array(f['3D'])
    mzrange = np.array(f['mzrange'])

    # print(r2d.shape, r3d.shape)
    #

    spectra_peak_picked, peakmzs = ImzObj.peak_pick(r2d, refmz=mzrange)
    mzs, ints = map(lambda x: np.asarray(x), ImzObj.parser.getspectrum(15))
    print(type(mzs), type(ints), type(r2d), type(mzrange), "\n", type(spectra_peak_picked),
          type(peakmzs))

    # plt.plot(peakmzs, spectra[1593, :])
    # plt.show()
    #

    # plt.subplot(211)
    # plt.plot(mrange, r2d[15, :])
    # # plt.show()
    # plt.subplot(212)
    # plt.plot(peakmzs, spectra[15, :])
    # plt.show()
    # plt.stem()
    # fig, vax = plt.subplots(figsize=(12, 6))
    # vax.vlines(np.array(peakmzs), [0], spectra_peak_picked[2093, :], color=(1, 0, 0), alpha=0.5)
    # plt.vlines(, , 0, 1, transform=vax.get_xaxis_transform())
    # plt.show()
    from sklearn.preprocessing import MinMaxScaler
    from Utilities import normalize_spectrum
    mms = MinMaxScaler()
    nS = 3135
    abraw = normalize_spectrum(r2d[nS, :], normalize='tic')
    abpro = normalize_spectrum(spectra_peak_picked[nS, :], normalize='tic')
    rawVSprocessed(mzrange, abraw,
                   np.array(peakmzs), abpro, exprun='resmapling+peak-picking')
    array3D, array2D, longestmz, regionshape, lCoorIdx = ImzObj.get_region(regID)
    peak3D = np.zeros([regionshape[0], regionshape[1], len(peakmzs)])
    for idx, coord in enumerate(lCoorIdx):
        peak3D[coord[0], coord[1], :] = spectra_peak_picked[idx, :]

    peakvar = []
    for mz in range(len(peakmzs)):
        peakvar.append(np.std(peak3D[..., mz]))

    # plt.plot(peakvar)
    # plt.show()

    peakvarHigh_Low = sorted(peakvar, reverse=True)
    # max_value = max(peakvar)
    plt.plot(peakvarHigh_Low)
    plt.show()
    cdict2 = {'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 1.0),
                      (1.0, 0.1, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.1),
                       (0.5, 1.0, 0.0),
                       (1.0, 0.0, 0.0))
              }

    cdict3 = {'red': ((0.0, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.5, 0.8, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 0.4, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (0.25, 0.0, 0.0),
                        (0.5, 0.9, 0.9),
                        (0.75, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.4),
                       (0.25, 1.0, 1.0),
                       (0.5, 1.0, 0.8),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }
    colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
    n_bin = 100
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    # mpl.colormaps.register(LinearSegmentedColormap('BlueRed2', cdict2))
    # mpl.colormaps.register(LinearSegmentedColormap('BlueRed3', cdict3))
    mpl.colormaps.register(LinearSegmentedColormap.from_list(name='simple_list', colors=colors, N=n_bin))
    topN = 50
    topmzInd = sorted(sorted(range(len(peakvar)), reverse=False, key=lambda sub: peakvar[sub])[-topN:])
    # for pv in peakvarHigh_Low[0:10]:
    #     plt.imshow(peak3D[..., peakvar.index(pv)].T, origin='lower', cmap='simple_list')
    #     plt.colorbar()
    #     mz = peakmzs[peakvar.index(pv)]
    #     plt.title("{}".format(mz))
    #     plt.show()

    # for pv in topmzInd:
    #     plt.imshow(peak3D[..., pv].T, origin='lower', cmap='simple_list')
    #     plt.colorbar()
    #     mz = peakmzs[pv]
    #     plt.title("{}".format(mz))
    #     plt.show()

    # peakvarLow_High = sorted(peakvar)
    # plt.plot(peakvarLow_High)
    # plt.show()

    # for mz in peakvarLow_High[200:210]:
    #     plt.imshow(peak3D[..., peakvar.index(mz)].T, cmap='simple_list')
    #     plt.colorbar()
    #     plt.title("{}".format(peakmzs[peakvar.index(mz)]))
    #     plt.show()
    if __name__ != '__main__':
        Nr = 10
        Nc = 5
        heights = [regionshape[1] for r in range(Nr)]
        widths = [regionshape[0] for r in range(Nc)]
        print(heights, widths)

        fig_width = 5.  # inches
        fig_height = fig_width * sum(heights) / sum(widths)
        fig, axs = plt.subplots(Nr, Nc, figsize=(fig_width, fig_height), dpi=600, constrained_layout=True,
                                gridspec_kw={'height_ratios': heights})
        fig.suptitle('reg {}: ion images'.format(regID), y=0.99)

        images = []
        pv = 0
        for r in range(Nr):
            for c in range(Nc):
                # Generate data with a range that varies from one plot to the next.
                # data = ((1 + i + j) / 10) * np.random.rand(10, 20)
                images.append(axs[r, c].imshow(peak3D[..., topmzInd[pv]].T, origin='lower', cmap='simple_list')) # 'RdBu_r')) #
                axs[r, c].label_outer()
                axs[r, c].set_axis_off()
                axs[r, c].set_title('{}'.format(peakmzs[topmzInd[pv]]), fontsize=5, pad=0.25)
                # fig.set_tight_layout('tight')
                # plt.gca().set_axis_off()
                fig.subplots_adjust(top=0.95, bottom=0.02, left=0,
                                    right=1, hspace=0.14, wspace=0)
                # fig.margins(0, 0)
                # fig.gca().xaxis.set_major_locator(plt.NullLocator())
                # fig.gca().yaxis.set_major_locator(plt.NullLocator())
                pv += 1

        # from matplotlib import colors
        # Find the min and max of all colors for use in setting the color scale.
        # vmin = min(image.get_array().min() for image in images)
        # vmax = max(image.get_array().max() for image in images)
        # norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # for im in images:
        #     im.set_norm(colors)

        # fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.2)


        # Make images respond to changes in the norm of other images (e.g. via the
        # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
        # recurse infinitely!
        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                        or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())
                    im.set_tight_layout('tight')

        for im in images:
            im.callbacks.connect('changed', update)

        # fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0,
        #                                 wspace=0)
        # fig.tight_layout('tight')
        # fig.set_axis_off()
        # fig.tight_layout()
        plt.show()
    nS = np.random.randint(spectra_peak_picked.shape[0])
    print(spectra_peak_picked.shape, nS)
    print(lCoorIdx[nS])
    # plt.plot()

    import matplotlib.collections as collections

    t = np.arange(0.0, 2, 0.01)
    s1 = peak3D[lCoorIdx[nS][0], lCoorIdx[nS][1], :]#np.sin(2 * np.pi * t)
    s2 = 1.2 * np.sin(4 * np.pi * t)
    # print("s2>0", s2>0)
    fig, ax = plt.subplots(figsize=(8, 8))
    # plt.ion()
    # ax.set_title('using span_where')
    markerline, stemlines, baseline = ax.stem(
        peakmzs, s1, linefmt='cyan', markerfmt="", basefmt="", use_line_collection=True) #
    # markerline.set_markerfacecolor('none')
    # ax.axhline(0, color='black', lw=2)
    for t in topmzInd:
        print(t)
        ax.axvspan(peakmzs[t]-0.02, peakmzs[t]+0.02, zorder=0.6, facecolor='blue')#, alpha=0.5)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
    # collection = collections.BrokenBarHCollection.span_where(
    #     np.array(peakmzs), ymin=0, ymax=max(s1), where=[topmzInd], facecolor='black', alpha=0.5)
    # ax.add_collection(collection)

    # collection = collections.BrokenBarHCollection.span_where(
    #     t, ymin=-1, ymax=0, where=s1 < 0, facecolor='red', alpha=0.5)
    # ax.add_collection(collection)
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    # Make the zoom-in plot:
    x1 = 595
    x2 = 610
    y1 = 0
    y2 = 1e6 #max(s1)
    axins = zoomed_inset_axes(ax, zoom=30, loc=1)#, bbox_transform=ax.figure.transFigure)  # zoom = 2
    axins.plot(peakmzs, s1)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.draw()
    plt.show()

# +-----------------------+
# |     ion image         |
# +-----------------------+
if __name__ != '__main__':
    ImzObj = ImzmlAll(mspathList[0])
    # print(ImzObj.parser.mzLengths)
    dict_ = ImzObj.parser.imzmldict
    print(dict_)
    # ImzObj._get_regions()
    im = getionimage(ImzObj.parser, mz_value=800, tol=0.02)
    plt.imshow(im)
    plt.colorbar()
    plt.show()
    # minmz = np.inf
    # maxmz = -np.inf
    # for s in range(len(ImzObj.parser.mzLengths)):
    #     minmz = min(ImzObj.parser.getspectrum(s)[0][0], minmz)
    #     maxmz = max(ImzObj.parser.getspectrum(s)[0][-1], maxmz)
    # print(minmz, maxmz)
    # tol = 0.01
    # massrange = np.arange(minmz, maxmz, tol)
    # print(massrange.shape, "massrange")
    # mzs, ints = ImzObj.parser.getspectrum(153)
    # print(ints, mzs)
    regID = 1
    tol = 0.02
    r2d, r3d, mrange = ImzObj.resample_region(regID, tol=tol)
    print(r2d.shape, r3d.shape)
    regname = os.path.join(os.path.dirname(mspathList[0]), 'reg_{}_tol_{}.h5'.format(regID, tol))
    with h5py.File(regname, 'w') as pfile:
        pfile['2D'] = r2d
        pfile['3D'] = r3d
        pfile['mzrange'] = mrange
        # pfile['peakmzs'] = peakmzs

    # mzs, ints = map(lambda x: np.asarray(x), p.getspectrum(i))
    # print("ok", np.sum(np.where(mzs<mz_value-tol, 1, 0)), np.sum(np.where(mzs<mz_value+tol, 1, 0)))
    # from bisect import bisect_left, bisect_right
    # def _bisect_spectrum(mzs, mz_value, tol):
    #     ix_l, ix_u = bisect_left(mzs, mz_value - tol), bisect_right(mzs, mz_value + tol) - 1
    #     if ix_l == len(mzs):
    #         return len(mzs), len(mzs)
    #     if ix_u < 1:
    #         return 0, 0
    #     if ix_u == len(mzs):
    #         ix_u -= 1
    #     if mzs[ix_l] < (mz_value - tol):
    #         ix_l += 1
    #     if mzs[ix_u] > (mz_value + tol):
    #         ix_u -= 1
    #     return ix_l, ix_u
    # int_ = np.zeros_like(massrange)
    # for idx, mz_value in enumerate(massrange):
    #     min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
    #     # print(">>",min_i, max_i+1, reduce_func(ints[min_i:max_i+1]), "\n")
    #     int_[idx] = sum(ints[min_i:max_i + 1])
    # # print(idx)

    # masses = ImzObj.parser.getspectrum(1593)[0]
    # spec_new = ImzObj._interpolate_spectrum(ints, mzs, massrange, method="Pchip")
    # rawVSprocessed(mzs, ints, massrange, int_, labels=['old data', 'corrected'],
    #                exprun='{} tolerance'.format(tol))

    # rawVSprocessed(mzs, ints, massrange, spec_new, labels=['old data', 'corrected'],
    #                exprun='{} tolerance'.format(tol))
    # intensity = np.zeros(len(massrange)-1)
    # for i in range(len(massrange)-1):
    #     intensity[i] = sum()

    # plt.subplot(2, 1, 1)
    # plt.plot(masses, spec)
    # plt.subplot(2, 1, 2)
    # plt.plot(massrange, spec_new)
    # plt.show()

    # np.median()
    # for spectra in range(len(ImzObj.parser.mzLengths)):
# (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = ImzObj.get_region_range(2)
# print((minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx)
# array3D, array2D, longestmz, regionshape, lCoorIdx=ImzObj.get_region(regID, whole=True)
# print(array2D.shape)
# spectra, peak_mz = ImzObj.peak_pick(, refmz2)
# +-----------------------+
# |   median check        |
# +-----------------------+
if __name__ != '__main__':
    medSpec = []
    ImzObj = ImzmlAll(mspathList[0])
    for s in range(len(ImzObj.parser.mzLengths)):
        medSpec.append(np.median(ImzObj.parser.getspectrum(s)[1]))
    # _boxplot(medSpec, labels=['old'])
    max_value = max(medSpec)
    min_value = min(medSpec)

    max_index = medSpec.index(max_value)
    min_index = medSpec.index(min_value)
    print(medSpec[max_index], max(medSpec))

    spec_max_med = ImzObj.parser.getspectrum(max_index)[1]
    spec_min_med = ImzObj.parser.getspectrum(min_index)[1]
    plt.plot(spec_max_med)
    plt.show()

    plt.plot(spec_min_med)
    plt.show()

# +-----------------------+
# |   processing not ok   |
# +-----------------------+
if __name__ != '__main__':
    ImzObj = ImzmlAll(mspath)
    regID = 2
    # meanSpec = ImzObj.get_mean_abundance()
    spec3D, spectra, refmz, regionshape, localCoor = ImzObj.get_region(regID, whole=False)
    print(spectra.shape)
    smooth_spectra = ImzObj.smooth_spectra(spectra, window_length=9)
    peak_spectra, peak_mz = ImzObj.peak_pick(smooth_spectra, refmz)   #, meanSpec)
    print(peak_spectra.shape)
    rawVSprocessed(refmz, spectra[1000], peak_mz, peak_spectra[1000])

# +------------------------------------+
# |   check max vs med interpolation   |
# +------------------------------------+
if __name__ != '__main__':
    ImzObj = ImzmlAll(mspathList[0])
    ImzObj1 = ImzmlAll(mspathList[1])

    minmz = min(ImzObj.parser.mzLengths)
    maxmz = max(ImzObj.parser.mzLengths)
    medmz = np.median(ImzObj.parser.mzLengths)
    print(minmz, maxmz, medmz)
    max_index = ImzObj.parser.mzLengths.index(maxmz)
    min_index = ImzObj.parser.mzLengths.index(minmz)
    med_index = np.round(ImzObj.parser.mzLengths.index(medmz))
    print(min_index, max_index, med_index)
    masses = ImzObj.parser.getspectrum(max_index)[0]
    # masses_new = ImzObj.parser.getspectrum(med_index)[0]
    spec = ImzObj.parser.getspectrum(max_index)[1]

    minmz1 = min(ImzObj1.parser.mzLengths)
    maxmz1 = max(ImzObj1.parser.mzLengths)
    medmz1 = np.median(ImzObj1.parser.mzLengths)
    max_index1 = ImzObj1.parser.mzLengths.index(maxmz1)
    min_index1 = ImzObj1.parser.mzLengths.index(minmz1)
    med_index1 = np.round(ImzObj1.parser.mzLengths.index(medmz1))
    masses1 = ImzObj1.parser.getspectrum(max_index1 - 10)[0]
    spec1 = ImzObj1.parser.getspectrum(max_index1 - 10)[1]

    spec_new = ImzObj._interpolate_spectrum(spec, masses, masses_new, method="Pchip")


    # plt.plot(ImzObj.parser.mzLengths)
    # plt.show()


    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    def baseline_als1(y, lam, p, niter=100):
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    plt.plot(masses1, spec1)
    plt.show()

    # spec_ = baseline_als1(spec1, lam=1e4, p=0.01)


    def baseline_als2(y, lam, p, niter=100):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z


    def baseline_als_optimized(y, lam, p, niter=100):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    from numpy.linalg import norm
    def baseline_arPLS(y, ratio=1e-6, lam=1e5, niter=100, full_output=False):
        L = len(y)

        diag = np.ones(L - 2)
        D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)

        H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        crit = 1
        count = 0

        while crit > ratio:
            z = spsolve(W + H, W * y)
            d = y - z
            dn = d[d < 0]

            m = np.mean(dn)
            s = np.std(dn)

            w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

            crit = norm(w_new - w) / norm(w)

            w = w_new
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values

            count += 1

            if count > niter:
                print('Maximum number of iterations exceeded')
                break

        if full_output:
            info = {'num_iter': count, 'stop_criterion': crit}
            return z, d, info
        else:
            return z

    # spec_ = baseline_als_optimized(spec1, lam=1e4, p=0.01)
    # spec_, spectra_arPLS, info = baseline_arPLS(spec1, lam=1e4, niter=100,
    #                                         full_output=True)
    # spectra_arPLS[spectra_arPLS<0]=0
    # m_ = np.round(np.median(spec1))

    def med_base_correct(spec1):
        m_ = np.round(np.median(spec1))
        spec_ = spec1
        while m_ > 2500:
            spec_ = spec_ - m_
            spec_[spec_ < 0] = 0
            nzInd = np.nonzero(spec_)
            m_ = np.round(np.median(spec_[nzInd]))
        return spec_, m_

    spmd, md = med_base_correct(spec1)
    print("md", md)
    # print(m_)
    # print(spec1)
    # spec_ = spec1 - m_
    # spec_[spec_<0]=0
    # nzInd = np.nonzero(spec_)
    # spec_ = spec_ - np.round(np.median(spec_[nzInd]))
    # spec_[spec_ < 0] = 0

    rawVSprocessed(masses1, spec1, masses1, spmd, labels=['flat off', 'corrected'], exprun='220210_reyzerml_IMC_380_plate1A baseline ')
    # print(len(spec_new))

# +----------------+
# |   old vs new   |
# +----------------+
if __name__ != '__main__':
    labels = ['old', 'new1', 'new2', 'new3', 'new4']
    allImzs = []
    for i in mspathList:
        # mspath = glob(os.path.join(i, '*.imzML'))[0]
        print(i)
        ImzObj = ImzmlAll(i)
        # print(len(ImzObj.parser.coordinates), ImzObj.parser.coordinates)
        # break
        medianIntensities = []
        for s in range(len(ImzObj.parser.coordinates)):
            medianIntensities.append(np.median(ImzObj.parser.getspectrum(s)[1]))
        allImzs.append(medianIntensities)
    # mspath = glob(os.path.join(posLip, '*.imzML'))[0]
    # print(mspath)
    # mspathNew = glob(os.path.join(posLipNew, '*.imzML'))[0]
    # print(mspathNew)
    # ImzObj = ImzmlAll(mspath)
    # ImzObjNew = ImzmlAll(mspathNew)
    # print(len(ImzObj.parser.coordinates))
    # print(len(ImzObjNew.parser.coordinates))
    # for i in range(Imz)
    # rawVSprocessed()
    # minmz = []
    # maxmz = []
    # for i in range(len(ImzObj.parser.coordinates)):
    #     minmz.append(ImzObj.parser.getspectrum(i)[0][0])
    #     maxmz.append(ImzObj.parser.getspectrum(i)[0][-1])
    #
    # plt.plot(minmz)
    # plt.plot(maxmz)
    # plt.show()

    # minmz2 = []
    # maxmz2 = []
    # for i in range(len(ImzObjNew.parser.coordinates)):
    #     minmz2.append(ImzObjNew.parser.getspectrum(i)[0][0])
    #     maxmz2.append(ImzObjNew.parser.getspectrum(i)[0][-1])
    #
    # plt.plot(minmz2)
    # plt.plot(maxmz2)
    # plt.show()
    #
    # plt.plot(ImzObj.parser.mzLengths)
    # plt.title("mzlengths in old")
    # plt.show()
    #
    # print("min - max mz old", min(ImzObj.parser.mzLengths), max(ImzObj.parser.mzLengths))
    #
    # plt.plot(ImzObjNew.parser.mzLengths)
    # plt.title("mzlengths in new")
    # plt.show()
    #
    # print("min - max mz new", min(ImzObjNew.parser.mzLengths), max(ImzObjNew.parser.mzLengths))
    #
    # # Observation:
    # # 1. the new data looks like have same range of m/zs like old lipid data.
    # # 2. bad interpolation in new images... why?
    # #         a. reference m/z bins too high?
    # #            min - max mz old 17255 36996
    # #            min - max mz new 17728 43668
    # #
    # # regID = 1
    # # array3D, array2D, longestmz, regionshape, lCoorIdx = ImzObjNew.get_region(regID=regID, whole=False)
    # # (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = ImzObjNew.get_region_range(regID=regID, whole=False)
    # # print("mzidx >>", (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx)
    # #
    # # array3D, array2D, longestmz, regionshape, lCoorIdx = ImzObj.get_region(regID=regID, whole=False)
    #
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)
    # labels = ['old', 'new']
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
    bplot = ax1.boxplot(list(allImzs),
                        notch=False, sym='+', vert=True,
                        patch_artist=True, whis=1.5, labels=labels,
                        medianprops=medianprops, meanprops=meanlineprops,
                        showfliers=True, showmeans=True, meanline=True)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='gray',
                   alpha=0.6)
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='',
        xlabel='Positive Lipids',
        ylabel='Value')
    ax1.set_title('Median intensity distribution', fontsize=16)
    # colors = ['maroon', 'darkblue', 'orangered', 'olive', 'sienna']
    # colors = ['maroon', 'darkblue', 'orangered', 'olive', 'sienna']
    medians = [bplot['medians'][i].get_ydata()[0] for i in range(len(labels))]
    pos = np.arange(len(labels)) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    for tick, label in zip(range(len(labels)), ax1.get_xticklabels()):
        ax1.text(pos[tick], .97, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', fontsize=10,
                 weight='bold',
                 color='firebrick') #colors[tick])
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    for patch in bplot['boxes']: #, colors):
        patch.set_facecolor('slategrey')
    fig.show()

if __name__ != '__main__':
    ImzObj = ImzmlAll(mspath)
    array, regs = ImzObj._get_regions()
    plt.imshow(array) #.T)
    plt.colorbar()
    plt.show()
    # print(regs)
    regID = 1
    print(ImzObj.get_region_range(regID, whole=False))
    print(ImzObj.get_region_range(regID, whole=True))
    # for i in range(regs):
    #     print("processing region: ", i+1)
    msmlfunc4(mspath, regID=regID, threshold=0.95, exprun='for_ANOVA3')

# print(len(ImzObj.parser.mzLengths), len(ImzObj.parser.coordinates))
# spectralength = 0
# mzidx = 0
# for sidx, coor in enumerate(ImzObj.parser.coordinates):
#     # print(sidx, coor)
#     if ImzObj.parser.mzLengths[sidx] > spectralength:
#         mzidx = sidx
#         spectralength = ImzObj.parser.mzLengths[mzidx]
# print(mzidx, ImzObj.parser.mzLengths[mzidx])
#
# spectra2D = [] #np.zeros()q
# for sidx, coor in enumerate(ImzObj.parser.coordinates):
#     spectra = ImzObj.parser.getspectrum(sidx)
#     # print(np.array(spectra).shape)
#     interp_spectra = ImzObj._interpolate_spectrum(spectra[1], spectra[0], ImzObj.parser.getspectrum(mzidx)[0])
#     spectra2D.append(interp_spectra)
#     # break
# print(np.array(spectra2D, dtype=np.float32).shape)
#
# from Utilities import rawVSprocessed
#
# rawVSprocessed(ImzObj.parser.getspectrum(546)[0], ImzObj.parser.getspectrum(546)[1],
#                ImzObj.parser.getspectrum(mzidx)[0], spectra2D[546])
#
# (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = ImzObj.get_region_range(2)
# regionshape = [maxx - minx + 1,
#                maxy - miny + 1]
# regionshape.append(ImzObj.parser.mzLengths[mzidx])
# array3D_2 = np.zeros(regionshape, dtype=np.float32)

# print(ImzObj.get_region_range(2, whole=True))
# print(ImzObj.get_region_range(2, whole=False))

if __name__ != '__main__':
    ImzObj = ImzmlAll(mspath)
    regID = 2
    spec3D2, spectra2, refmz2, regionshape2, localCoor2 = ImzObj.get_region(regID, whole=True)
    regID = 3
    spec3D3, spectra3, refmz3, regionshape3, localCoor3 = ImzObj.get_region(regID, whole=True)

    print(spec3D2.shape)
    print("2d array 2 >>", spectra2.shape)
    print(spec3D3.shape)
    print("2d array 3 >>", spectra3.shape)
# +--------------------+
# |        ANOVA       |
# +--------------------+
if __name__ != '__main__':
    ImzObj = ImzmlAll(mspath)
    regID = 1
    spec3D1, spectra1, refmz1, regionshape1, localCoor1 = ImzObj.get_region(regID, whole=True)
    print("len(localCoor1) >>", len(localCoor1))
    regID = 2
    spec3D2, spectra2, refmz2, regionshape2, localCoor2 = ImzObj.get_region(regID, whole=True)
    regID = 3
    spec3D3, spectra3, refmz3, regionshape3, localCoor3 = ImzObj.get_region(regID, whole=True)
    regID = 4
    spec3D4, spectra4, refmz4, regionshape4, localCoor4 = ImzObj.get_region(regID, whole=True)
    regID = 5
    spec3D5, spectra5, refmz5, regionshape5, localCoor5 = ImzObj.get_region(regID, whole=True)

    print("3d reg 1", spec3D1.shape)
    print("2d reg 1 >>", spectra1.shape)
    print("3d reg 2", spec3D2.shape)
    print("2d reg 2 >>", spectra2.shape)
    print("3d reg 3", spec3D3.shape)
    print("2d reg 3 >>", spectra3.shape)
    print("3d reg 4", spec3D4.shape)
    print("2d reg 4 >>", spectra4.shape)
    print("3d reg 5", spec3D5.shape)
    print("2d reg 5 >>", spectra5.shape)

    seg1 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_1', '*4_1.npy'))[0]
    seg2 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_2', '*4_1.npy'))[0]
    seg3 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_3', '*4_1.npy'))[0]
    seg4 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_4', '*4_1.npy'))[0]
    seg5 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_5', '*4_1.npy'))[0]

    label1 = np.load(seg1)
    label2 = np.load(seg2)
    label3 = np.load(seg3)
    label4 = np.load(seg4)
    label5 = np.load(seg5)
    # print(np.unique(label2))
    # print(np.unique(label3))
    elements, counts = np.unique(label1, return_counts=True)
    print("label1 >>", elements, counts)
    elements, counts = np.unique(label2, return_counts=True)
    print("label2 >>", elements, counts)
    elements, counts = np.unique(label3, return_counts=True)
    print("label3 >>", elements, counts)
    elements, counts = np.unique(label4, return_counts=True)
    print("label4 >>", elements, counts)
    elements, counts = np.unique(label5, return_counts=True)
    print("label5 >>", elements, counts)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    seg_arrays = [label1, label2, label3, label4, label5]
    title = ['reg1', 'reg2', 'reg3', 'reg4', 'reg5']
    butterfly_labels = [1, 2, 1, 2, 2]      # butterfly
    peripheral_labels = [3, 3, 2, 3, 1]     # peripheral
    gm_labels = [2, 1, 4, 1, 3]     # u-shape
    rest = [4, 4, 3, 4, 4]

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax in zip(seg_arrays, title, axs.ravel()):
        im = ax.imshow(ar)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), butterfly_labels):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), peripheral_labels):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), gm_labels):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), rest):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()
    # print(len(np.where(label1 == 1)[0]))
    # print(type(localCoor), "\n", localCoor)

    Fvalues = []
    pvalues = []
    logpvalues = []

    for _labels in (butterfly_labels, peripheral_labels, gm_labels, rest):
        print("_labels >> ", _labels)
        _labels = list(_labels)
        spec_lab_1 = []
        spec_lab_2 = []  # np.zeros([len(np.where(label1 == 4)[0]), spectra.shape[1]], dtype=np.float32)
        spec_lab_3 = []  # np.zeros([len(np.where(label1 == 3)[0]), spectra.shape[1]], dtype=np.float32)
        spec_lab_4 = []
        spec_lab_5 = []

        img = np.zeros([regionshape1[0], regionshape1[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label1 == _labels[0])[0], np.where(label1 == _labels[0])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor1):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_lab_1.append(spectra1[ldx, :])  # [idx, :] = spectra[ldx, :]
        plt.imshow(img)
        plt.show()

        img = np.zeros([regionshape2[0], regionshape2[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label2 == _labels[1])[0], np.where(label2 == _labels[1])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor2):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_lab_2.append(spectra2[ldx, :])  #[idx, :] = spectra[ldx, :]
        plt.imshow(img)
        plt.show()

        img = np.zeros([regionshape3[0], regionshape3[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label3 == _labels[2])[0], np.where(label3 == _labels[2])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor3):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_lab_3.append(spectra3[ldx, :])
                    # spec_lab_4[idx, :] = spectra[ldx, :]
        plt.imshow(img)
        plt.show()
        img = np.zeros([regionshape4[0], regionshape4[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label4 == _labels[3])[0], np.where(label4 == _labels[3])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor4):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_lab_4.append(spectra4[ldx, :])
        plt.imshow(img)
        plt.show()

        img = np.zeros([regionshape5[0], regionshape5[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label5 == _labels[4])[0], np.where(label5 == _labels[4])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor5):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_lab_5.append(spectra5[ldx, :])
        plt.imshow(img)
        plt.show()

        spec_lab_1 = np.array(spec_lab_1)
        spec_lab_2 = np.array(spec_lab_2)
        spec_lab_3 = np.array(spec_lab_3)
        spec_lab_4 = np.array(spec_lab_4)
        spec_lab_5 = np.array(spec_lab_5)
        print("b4 1-way >>", spec_lab_2.shape, spec_lab_5.shape)
        # fvalue, pvalue = stats.f_oneway(np.mean(spec_lab_2, axis=0), np.mean(spec_lab_3, axis=0), axis=0)
        # fvalue, pvalue = stats.f_oneway(spec_lab_3[0:1000, :], spec_lab_3[1000:, :]) #, axis=0)
        Fvalue, pvalue = stats.f_oneway(spec_lab_1, spec_lab_1, spec_lab_1, spec_lab_1, spec_lab_1)  # , axis=0)
        # Fvalue, pvalue = stats.f_oneway(spec_lab_1, spec_lab_1, spec_lab_1, spec_lab_1, spec_lab_1)  # all same test
        # print(spec_lab_3)#.shape, spec_lab_4.shape)
        Fvalues.append(Fvalue)
        pvalues.append(pvalue)
        print("F >>", Fvalue)
        # print(len(fvalue), "\n")
        print("p >>", pvalue, np.sum(pvalue), np.mean(pvalue))
        # print(len(pvalue))
        plt.plot(Fvalue)
        plt.title("F value")
        plt.show()
        plt.plot(pvalue)
        plt.title("p-value")
        plt.show()
        # break
        pvalue_log = -np.log10(pvalue)
        logpvalues.append(pvalue_log)
        plt.plot(pvalue_log)
        plt.title("-log10(p-value)")
        plt.show()
        break
    dANOVA = {'Fvalue': Fvalues,
              'pvalue': pvalues,
              'logpvalue': logpvalues
             }
    aPath = os.path.join(os.path.dirname(mspath), 'ANOVA_results_all5.bin')
    # print(mzPath)
    with open(aPath, 'wb') as pfile:
        pickle.dump(dANOVA, pfile)

    p_values = pd.DataFrame(pvalues)
    ax = sns.boxplot(data=p_values.T)
    plt.show()
    Fvalues = pd.DataFrame(Fvalues)
    ax = sns.boxplot(data=Fvalues.T)
    plt.show()
    logp_values = pd.DataFrame(logpvalues)
    ax = sns.boxplot(data=logp_values.T)
    plt.title("-log10(p-value)")
    plt.show()

# +---------------------------+
# |       ANOVA regions       |
# +---------------------------+
if __name__ != '__main__':
    ImzObj = ImzmlAll(mspathList[0])
    regID = 1
    spec3D1, spectra1, refmz1, regionshape1, localCoor1 = ImzObj.get_region(regID, whole=True)
    print("len(localCoor1) >>", len(localCoor1))
    regID = 2
    spec3D2, spectra2, refmz2, regionshape2, localCoor2 = ImzObj.get_region(regID, whole=True)
    regID = 3
    spec3D3, spectra3, refmz3, regionshape3, localCoor3 = ImzObj.get_region(regID, whole=True)
    regID = 4
    spec3D4, spectra4, refmz4, regionshape4, localCoor4 = ImzObj.get_region(regID, whole=True)
    regID = 5
    spec3D5, spectra5, refmz5, regionshape5, localCoor5 = ImzObj.get_region(regID, whole=True)

    print("3d reg 1", spec3D1.shape)
    print("2d reg 1 >>", spectra1.shape)
    print("3d reg 2", spec3D2.shape)
    print("2d reg 2 >>", spectra2.shape)
    print("3d reg 3", spec3D3.shape)
    print("2d reg 3 >>", spectra3.shape)
    print("3d reg 4", spec3D4.shape)
    print("2d reg 4 >>", spectra4.shape)
    print("3d reg 5", spec3D5.shape)
    print("2d reg 5 >>", spectra5.shape)

    seg1_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_1', '*4_1.npy'))[0]
    seg2_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_2', '*4_1.npy'))[0]
    seg3_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_3', '*4_1.npy'))[0]
    seg4_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_4', '*4_1.npy'))[0]
    seg5_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_5', '*4_1.npy'))[0]

    seg_reg_1 = np.load(seg1_path)
    seg_reg_2 = np.load(seg2_path)
    seg_reg_3 = np.load(seg3_path)
    seg_reg_4 = np.load(seg4_path)
    seg_reg_5 = np.load(seg5_path)
    # print(np.unique(label2))
    # print(np.unique(label3))
    elements, counts = np.unique(seg_reg_1, return_counts=True)
    print("label1 >>", elements, counts)
    elements, counts = np.unique(seg_reg_2, return_counts=True)
    print("label2 >>", elements, counts)
    elements, counts = np.unique(seg_reg_3, return_counts=True)
    print("label3 >>", elements, counts)
    elements, counts = np.unique(seg_reg_4, return_counts=True)
    print("label4 >>", elements, counts)
    elements, counts = np.unique(seg_reg_5, return_counts=True)
    print("label5 >>", elements, counts)

    seg_arrays = [seg_reg_1, seg_reg_2, seg_reg_3, seg_reg_4, seg_reg_5]
    title = ['reg1', 'reg2', 'reg3', 'reg4', 'reg5']
    label_names = ['wm(butterfly)', 'peripheral', 'gm(u-shape)', 'others']
    butterfly_labels = [1, 2, 1, 2, 2]      # butterfly
    peripheral_labels = [3, 3, 2, 3, 1]     # peripheral
    gm_labels = [2, 1, 4, 1, 3]     # u-shape
    rest = [4, 4, 3, 4, 4]

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax in zip(seg_arrays, title, axs.ravel()):
        im = ax.imshow(ar)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), butterfly_labels):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), peripheral_labels):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), gm_labels):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(seg_arrays, title, axs.ravel(), rest):
        im = ax.imshow(ar==bl)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()
    # print(len(np.where(label1 == 1)[0]))
    # print(type(localCoor), "\n", localCoor)

    Fvalues = []
    pvalues = []
    logpvalues = []

    for hdx, _labels in enumerate([butterfly_labels, peripheral_labels, gm_labels, rest]):
        print("_labels >> ", _labels)
        _labels = list(_labels)
        spec_reg_1 = []
        spec_reg_2 = []  # np.zeros([len(np.where(label1 == 4)[0]), spectra.shape[1]], dtype=np.float32)
        spec_reg_3 = []  # np.zeros([len(np.where(label1 == 3)[0]), spectra.shape[1]], dtype=np.float32)
        spec_reg_4 = []
        spec_reg_5 = []

        img = np.zeros([regionshape1[0], regionshape1[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(seg_reg_1 == _labels[0])[0], np.where(seg_reg_1 == _labels[0])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor1):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_reg_1.append(spectra1[ldx, :])  # [idx, :] = spectra[ldx, :]
        plt.imshow(img)
        plt.show()

        img = np.zeros([regionshape2[0], regionshape2[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(seg_reg_2 == _labels[1])[0], np.where(seg_reg_2 == _labels[1])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor2):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_reg_2.append(spectra2[ldx, :])  #[idx, :] = spectra[ldx, :]
        plt.imshow(img)
        plt.show()

        img = np.zeros([regionshape3[0], regionshape3[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(seg_reg_3 == _labels[2])[0], np.where(seg_reg_3 == _labels[2])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor3):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_reg_3.append(spectra3[ldx, :])
                    # spec_lab_4[idx, :] = spectra[ldx, :]
        plt.imshow(img)
        plt.show()
        img = np.zeros([regionshape4[0], regionshape4[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(seg_reg_4 == _labels[3])[0], np.where(seg_reg_4 == _labels[3])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor4):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_reg_4.append(spectra4[ldx, :])
        plt.imshow(img)
        plt.show()

        img = np.zeros([regionshape5[0], regionshape5[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(seg_reg_5 == _labels[4])[0], np.where(seg_reg_5 == _labels[4])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor5):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_reg_5.append(spectra5[ldx, :])
        plt.imshow(img)
        plt.show()

        spec_reg_1 = np.array(spec_reg_1)
        spec_reg_2 = np.array(spec_reg_2)
        spec_reg_3 = np.array(spec_reg_3)
        spec_reg_4 = np.array(spec_reg_4)
        spec_reg_5 = np.array(spec_reg_5)
        print("b4 1-way >>", spec_reg_2.shape, spec_reg_5.shape)     # (1359, 36996) (1057, 36996)
        if __name__ == '__main__':
            spec_reg_1_mean = np.mean(spec_reg_1, axis=0)
            spec_reg_2_mean = np.mean(spec_reg_2, axis=0)
            spec_reg_3_mean = np.mean(spec_reg_3, axis=0)
            spec_reg_4_mean = np.mean(spec_reg_4, axis=0)
            spec_reg_5_mean = np.mean(spec_reg_5, axis=0)
            print("spec_reg_4_mean.shape >>", spec_reg_4_mean.shape)
            specDict = {}
            mean_dict = [spec_reg_1_mean, spec_reg_2_mean, spec_reg_3_mean, spec_reg_4_mean, spec_reg_5_mean]
            for r in range(len(mean_dict)):  # 0 is background
                spec = {"{}".format(r + 1): mean_dict[r]}
                specDict.update(spec)
            print(specDict.keys())

            from scipy.spatial.distance import cosine
            def cosineSim(spectrum1, spectrum2):
                return 1 - cosine(spectrum1,
                                  spectrum2)


            spec_df = pd.DataFrame(specDict)
            method_ = ['pearson', 'spearman', 'kendall', cosineSim]
            corr = spec_df.corr(method=method_[3])
            # if corr.
            grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
            fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
            ax = sns.heatmap(corr, ax=ax, annot=True,
                             vmin=0.9, vmax=1, center=0.5,
                             cbar_ax=cbar_ax,
                             cbar_kws={"orientation": "horizontal"},
                             cmap='nipy_spectral', #'Greys',  # "YlGnBu",
                             linewidths=0.5,
                             # xticklabels='vertical',
                             # yticklabels='horizontal',
                             square=False)

            ax.set_yticklabels(specDict.keys(), rotation=0)
            ax.set_xticklabels(specDict.keys(), rotation=90)
            ax.xaxis.tick_top()
            ax.set_title("{}".format(label_names[hdx]))
            fig.show()

        # break
        # fvalue, pvalue = stats.f_oneway(np.mean(spec_lab_2, axis=0), np.mean(spec_lab_3, axis=0), axis=0)
        # fvalue, pvalue = stats.f_oneway(spec_lab_3[0:1000, :], spec_lab_3[1000:, :]) #, axis=0)
        Fvalue, pvalue = stats.f_oneway(spec_reg_1, spec_reg_2, spec_reg_3, spec_reg_4, spec_reg_5)  # , axis=0)
        # Fvalue, pvalue = stats.f_oneway(spec_lab_1, spec_lab_1, spec_lab_1, spec_lab_1, spec_lab_1)  # all same test
        # print(spec_lab_3)#.shape, spec_lab_4.shape)
        Fvalues.append(Fvalue)
        pvalues.append(pvalue)
        print("F >>", Fvalue)
        # print(len(fvalue), "\n")
        print("p >>", pvalue, np.sum(pvalue), np.mean(pvalue))
        # print(len(pvalue))
        plt.plot(Fvalue)
        plt.title("F value")
        plt.show()
        plt.plot(pvalue)
        plt.title("p-value")
        plt.show()
        # break
        pvalue_log = -np.log10(pvalue)
        logpvalues.append(pvalue_log)
        plt.plot(pvalue_log)
        plt.title("-log10(p-value)")
        plt.show()
    dANOVA = {'Fvalue': Fvalues,
              'pvalue': pvalues,
              'logpvalue': logpvalues
             }
    aPath = os.path.join(os.path.dirname(mspathList[0]), 'ANOVA_results_all5.bin')
    # print(mzPath)
    # with open(aPath, 'wb') as pfile:
    #     pickle.dump(dANOVA, pfile)


    p_values = pd.DataFrame(pvalues)
    ax = sns.boxplot(data=p_values.T)
    plt.show()
    Fvalues = pd.DataFrame(Fvalues)
    ax = sns.boxplot(data=Fvalues.T)
    plt.show()
    logp_values = pd.DataFrame(logpvalues)
    ax = sns.boxplot(data=logp_values.T)
    plt.title("-log10(p-value)")
    plt.show()

# +--------------------+
# |    ANOVA results   |
# +--------------------+
if __name__ != '__main__':
    aPath = os.path.join(os.path.dirname(mspath), 'ANOVA_results_all5.bin')
    with open(aPath, 'rb') as pfile:
        dANOVA = pickle.load(pfile)
    print(dANOVA.keys())
    Fv = dANOVA['Fvalue']
    pv = dANOVA['pvalue']
    # print(np.isnan(pv))
    pv = np.nan_to_num(pv, nan=0.98) # why? if similar then p-value is closer to 1
    # print(np.isnan(pv))
    logpv = -np.log10(pv) #dANOVA['logpvalue']
    # print(np.shape(pv))  #(4, 36996)
    # print(logpv)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)
    labels = ['butterfly', 'peripheral', 'gm', 'rest']
    medianprops = dict(linestyle='-', linewidth=1.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=1.5, color='purple')
    bplot = ax1.boxplot(list(logpv),
                        notch=False, sym='+', vert=True,
                        patch_artist=True, whis=1.5, labels=labels,
                        medianprops=medianprops, meanprops=meanlineprops,
                        showfliers=True, showmeans=True, meanline=True)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='gray',
                   alpha=0.6)
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='',
        xlabel='Tissues',
        ylabel='Value')
    ax1.set_title('ANOVA: -log(p-value)', fontsize=16)
    colors = ['maroon', 'darkblue', 'orangered', 'olive']
    medians = [bplot['medians'][i].get_ydata()[0] for i in range(len(labels))]
    pos = np.arange(len(labels)) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    for tick, label in zip(range(len(labels)), ax1.get_xticklabels()):
        ax1.text(pos[tick], .97, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', fontsize=10,
                 weight='bold',
                 color=colors[tick])
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    fig.show()
# +--------------------+
# |    match regions   |
# +--------------------+
if __name__ != '__main__':
    # seg1_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_1', '*4_1.npy'))[0]
    # seg2_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_2', '*4_1.npy'))[0]
    # seg3_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_3', '*4_1.npy'))[0]
    # ImzObj = ImzmlAll(mspath)
    # regID = 1
    # spec3D1, spectra1, refmz1, regionshape1, localCoor1 = ImzObj.get_region(regID)
    # regID = 2
    # spec3D2, spectra2, refmz2, regionshape2, localCoor2 = ImzObj.get_region(regID)
    # regID = 3
    # spec3D3, spectra3, refmz3, regionshape3, localCoor3 = ImzObj.get_region(regID)
    # matchSpecLabel2(True, seg1_path, seg2_path, seg3_path, arr1=spec3D1, arr2=spec3D2, arr3=spec3D3, exprun='kinda whole')
    seg1_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_1', '*4_1.npy'))[0]
    seg2_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_2', '*4_1.npy'))[0]
    seg3_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_3', '*4_1.npy'))[0]
    seg4_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_4', '*4_1.npy'))[0]
    seg5_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_5', '*4_1.npy'))[0]
    ImzObj = ImzmlAll(mspath)
    regID = 1
    spec3D1, spectra1, refmz1, regionshape1, localCoor1 = ImzObj.get_region(regID, whole=True)
    # print("len(localCoor1) >>", len(localCoor1))
    regID = 2
    spec3D2, spectra2, refmz2, regionshape2, localCoor2 = ImzObj.get_region(regID, whole=True)
    regID = 3
    spec3D3, spectra3, refmz3, regionshape3, localCoor3 = ImzObj.get_region(regID, whole=True)
    regID = 4
    spec3D4, spectra4, refmz4, regionshape4, localCoor4 = ImzObj.get_region(regID, whole=True)
    regID = 5
    spec3D5, spectra5, refmz5, regionshape5, localCoor5 = ImzObj.get_region(regID, whole=True)
    matchSpecLabel2(True, seg1_path, seg2_path, seg3_path, seg4_path, seg5_path, arr1=spec3D1,
                                                                                arr2=spec3D2,
                                                                              arr3=spec3D3,
                                                                            arr4=spec3D4,
                                                                         arr5=spec3D5, exprun='whole mz')
    # print()
    # for l in range(1, len(np.unique(label1))):  # to avoid 0-background
    #     label1_ = copy.deepcopy(label1)
    #     label1_[label1 != l] = 0
    #     spec = np.mean(arr1[np.where(label1_)], axis=0)
    #     spec = {"{}_{}".format(1, l): spec}
    #     specDict.update(spec)
    #
    # for l in range(1, len(np.unique(label2))):  # to avoid 0-background
    #     label2_ = copy.deepcopy(label2)
    #     label2_[label2 != l] = 0
    #     spec = np.mean(arr2[np.where(label2_)], axis=0)
    #     spec = {"{}_{}".format(2, l): spec}
    #     specDict.update(spec)

# dirname = os.path.dirname(mspath)
# regDir = os.path.join(dirname, 'reg_{}'.format(regID))


# smooth_spectra = ImzObj.smooth_spectra(spectra, window_length=9, polyorder=2)
# # peak_spectra1, peakmzs = ImzObj.peak_pick(spectra, refmz)
# peak_spectra2, peakmzs = ImzObj.peak_pick(smooth_spectra, refmz)
# print(len(localCoor), regionshape)
# for idx, mz in enumerate(peakmzs):
#     i_img = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
#     for jdx, coor in enumerate(localCoor):
#         i_img[coor[0], coor[1]] = peak_spectra2[jdx, idx]
#     fname = os.path.join(regDir, 'ion_img_{}.png'.format(mz))
#     cv2.imwrite(fname, i_img)
#     break

    # plt.imshow(i_img)
    # plt.show()
    # break

# from Utilities import normalize_spectrum
# spec_norm = np.zeros_like(peak_spectra2)
# for s in range(peak_spectra2.shape[0]):
#     spec_norm[s, :] = normalize_spectrum(peak_spectra2[s, :], normalize='max_intensity_spectrum', max_region_value=None)
# plt.subplot(511)
# plt.plot(spectra[1200,...])
# plt.subplot(512)
# plt.plot(smooth_spectra[1200,...])
# plt.subplot(513)
# plt.plot(peak_spectra1[1200,...])
# plt.subplot(514)
# plt.plot(peak_spectra2[1200,...])
# plt.subplot(515)
# plt.plot(spec_norm[1200, ...])
# plt.show()
# print(peak_spectra1.shape)
# print(peak_spectra2.shape)



# print(regionshape[0], regionshape[1])
# nS = np.random.randint(spectra.shape[0])
# abraw = spectra[nS, :]

# abpro = spectra_[nS, :]
# _, reg_smooth_ = bestWvltForRegion(spectra, bestWvlt='db8', smoothed_array=True, plot_fig=True)
# from Utilities import rawVSprocessed
# rawVSprocessed(refmz, abraw, peakmzs, abpro)
# print(reg_smooth_.shape)



# msmlfunc3(mspath, regID=1, threshold=0.95, exprun='HC_ion_img', downsamp_i=None, wSize=None)

def regResamp(mspath, regID): #>> ImzObj,
    """
    saves intensities with same size vectors...
    """
    # dataDir = r''
    ImzObj = IMZMLExtract(mspath)
    mzList = []
    mzSum = 0
    regInd = ImzObj.get_region_indices(regID)
    mzPath = os.path.join(os.path.dirname(mspath), 'mzList_{}.bin'.format(regID))
    if not os.path.exists(mzPath):
        for coord in tqdm(regInd):
            # print(coord)
            spectrum = ImzObj.parser.getspectrum(ImzObj.coord2index.get(coord))  # [0]
            mzList += list(spectrum[0])
            mzSum += len(spectrum[0])
        print("mzList", len(mzList))
        print("mzSum", mzSum)
        elements, counts = np.unique(mzList, return_counts=True)

        mzDict = {'elements': elements,
                  'counts': counts}

        mzPath = os.path.join(os.path.dirname(mspath), 'mzList_{}.bin'.format(regID))
        # print(mzPath)
        with open(mzPath, 'wb') as pfile:
            pickle.dump(mzDict, pfile)
    else:
        with open(mzPath, 'rb') as pfile:
            mzDict = pickle.load(pfile)
        print(mzDict.keys())
        elements = mzDict['elements']
        counts = mzDict['counts']
    prcntl_thr = 95
    # print(np.percentile(counts, 95))
    # print(np.sum(counts <= np.percentile(counts, 95)))
    elements = elements[np.where(counts > np.percentile(counts, prcntl_thr))]
    # print("len(elements): ", len(elements))
    print("length of elements changed from {} to {}".format(len(counts), len(elements)))
    spec_data = np.zeros([len(ImzObj.get_region_indices(regID)), len(elements)])
    idx = 0
    for coord in tqdm(regInd):
        spectrum = ImzObj.parser.getspectrum(ImzObj.coord2index.get(coord))  # [0]
        spec_data[idx, :] = interpolate_spectrum(spectrum[1], spectrum[0], elements, method='Pchip')
        idx += 1

    interSpecPath = os.path.join(os.path.dirname(mspath), 'resSpec1D_{}.bin'.format(regID))
    # print(mzPath)
    with open(interSpecPath, 'wb') as pfile:
        pickle.dump(spec_data, pfile)

# regResamp(mspath, 1)
# +----------------------+
# |    get all bins      |
# +----------------------+
if __name__ != '__main__':
    mzList = []
    mzSum = 0
    for spx in tqdm(range(len(ImzObj.parser.intensityLengths))):
        sp = ImzObj.parser.getspectrum(spx)
        print(sp[0].dtype, type(sp[0]))
        mzSum += len(sp[0])
        # try:
        mzList += list(sp[0])
        break
    print("mzList", len(mzList))
    print("mzSum", mzSum)
    elements, counts = np.unique(mzList, return_counts=True)
    print("total m/z s {} and common m/z s {}".format(len(mzList), len(elements)))

    mzDict = {'mzList': mzList,
              'bins': elements}

    mzPath = os.path.join(posLip, 'mzList.bin')
    # print(mzPath)
    with open(mzPath, 'wb') as pfile:
        pickle.dump(mzDict, pfile)

# mzPath = os.path.join(posLip, 'mzList_1.bin')
# with open(mzPath, 'rb') as pfile:
#     mzDict = pickle.load(pfile)
# #
# print(mzDict.keys())
# # mzList = mzDict['mzList']
# elements = mzDict['elements']
# counts = mzDict['counts']

# regID = 1
# interSpecPath = os.path.join(os.path.dirname(mspath), 'resSpec1D_{}.bin'.format(regID))
# with open(interSpecPath, 'rb') as pfile:
#     resSpec1D = pickle.load(pfile)
#
# plt.plot(resSpec1D[2000, :])
# plt.show()
# print(resSpec1D.shape)
# ch_idx = 240
# plt.plot(resSpec1D[ch_idx])
# plt.show()
# ImzObj = IMZMLExtract(mspath)
# regInd = ImzObj.get_region_indices(regID)
# idx = 0
# for coord in tqdm(regInd):
#     spectrum = ImzObj.parser.getspectrum(ImzObj.coord2index.get(coord))
#     if idx==ch_idx:
#         plt.plot(spectrum[1])
#         plt.show()
#         break
#     idx+=1

if __name__ != '__main__':
    from Utilities import _smooth_spectrum,find_nearest, makeSS
    from sklearn.decomposition import PCA
    import pandas as pd
    import mglearn
    import matplotlib.cm as cm
    from sklearn.cluster import AgglomerativeClustering

    reg_norm = np.zeros_like(resSpec1D)
    for s in range(0, resSpec1D.shape[0]):
        reg_norm[s, :] = normalize_spectrum(resSpec1D[s, :], normalize='tic')  # reg_smooth_
        reg_norm[s, :] = _smooth_spectrum(reg_norm[s, :], method='savgol', window_length=5, polyorder=2)
    reg_norm_ss = makeSS(reg_norm).astype(np.float64)

    nS = np.random.randint(resSpec1D.shape[0])
    fig, ax = plt.subplots(4, 1, figsize=(16, 10), dpi=200)
    ax[0].plot(resSpec1D[nS, :])
    ax[0].set_title("raw spectrum")
    ax[1].plot(reg_norm[nS, :])
    ax[1].set_title("'tic' norm")
    ax[2].plot(reg_norm_ss[nS, :])
    ax[2].set_title("standardized")
    ax[3].plot(np.mean(resSpec1D, axis=0))
    ax[3].set_title("mean spectra(region {})".format(regID))
    # ax[4].plot(reg_smooth_[nS, :])
    # ax[4].set_title("Smoothed...")
    plt.suptitle("Processing comparison of Spec #{}".format(nS))
    plt.show()
    RandomState = 20210131
    pca = PCA(random_state=RandomState)     # pca object
    pcs = pca.fit_transform(reg_norm_ss)   # (4587, 2000)
    # pcs=pca.fit_transform(oldLipid_mm_norm)
    pca_range = np.arange(1, pca.n_components_, 1)
    print(">> PCA: number of components #{}".format(pca.n_components_))
    # printStat(pcs)
    evr = pca.explained_variance_ratio_
    # print(evr)
    evr_cumsum = np.cumsum(evr)
    # print(evr_cumsum)
    cut_evr = find_nearest(evr_cumsum, 0.95)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
    df_pca = pd.DataFrame(data=pcs[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])
    MaxPCs = nPCs + 5
    fig, ax = plt.subplots(figsize=(20, 8), dpi=200)
    ax.bar(pca_range[0:MaxPCs], evr[0:MaxPCs] * 100, color="steelblue")
    ax.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
    ax.set_xlabel('Principal component number', fontsize=30)
    ax.set_ylabel('Percentage of \n variance explained', fontsize=30)
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

    plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
    plt.show()

    nCl = 7
    agg = AgglomerativeClustering(n_clusters=nCl)
    assignment = agg.fit_predict(reg_norm_ss)  # on pca
    # mglearn.discrete_scatter(regCoor[:, 0], regCoor[:, 1], assignment, labels=np.unique(assignment))
    plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
    plt.title("Agglomerative Clustering")
    plt.show()

    plt.figure(figsize=(12, 10), dpi=200)
    # plt.scatter(df_pca.PC_1, df_pca.PC_2, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
    # plt.scatter(df_pca.PC_1, df_pca.PC_2, c=assignment, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
    mglearn.discrete_scatter(df_pca.PC_1, df_pca.PC_2, assignment, alpha=0.5) #, labels=np.unique(assignment))
    plt.xlabel('PC1 ({}%)'.format(round(evr[0] * 100, 2)), fontsize=30)
    plt.ylabel('PC2 ({}%)'.format(round(evr[1] * 100, 2)), fontsize=30)
    plt.tick_params(size=10, color='black')
    # tick and axis theme
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
    ax = plt.gca()  # Get the current Axes instance
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)
    plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
    plt.show()


# print("len(elements): {}, counts: {}".format(len(elements), counts))
#
# plt.plot(counts)
# plt.show()
# print(elements[0:100])
# print(np.percentile(counts, 95))
# print(np.sum(counts <= np.percentile(counts, 95)))
# elements = elements[np.where(counts > np.percentile(counts, 95))]
# print("len(elements): ", len(elements))
# print(elements)
# spec_data = np.zeros([len(ImzObj.get_region_indices(regID)), len(elements)]) ## (3435, 1332)





# spec_array, spec_data, coordList = Binning2(ImzObj, regID, n_bins=2000).MaldiTofBinning()
# dirname = os.path.dirname(mspath)
# basename = os.path.basename(mspath)
# filename, ext = os.path.splitext(basename)
# regDir = os.path.join(dirname, 'reg_{}'.format(regID))
# if not os.path.isdir(regDir):
#     os.mkdir(regDir)
# regname = os.path.join(regDir, '{}_reg_{}.mat'.format(filename, regID))
# matr = {"spectra": spec_data, "array": spec_array, "coordinates": coordList, "info": "unbinned"}
# savemat(regname, matr)
# for regID in [1, 2, 5]:
#     print("Processing: ", regID)
    # msmlfunc3(mspath, regID=regID, threshold=0.95, exprun='maldi-learn_14k_savgol', downsamp_i=None)
# regname = os.path.join(posLip, '{}_reg_{}.mat'.format(posLip, regID))
# BinObj = Binning2(ImzObj, 3)
# regArr, regSpec, spCoo = BinObj.getBinMat()
# reg1_path = r'C:\Data\PosLip\reg_1_ub'
# reg2_path = r'C:\Data\PosLip\reg_2_ub'
# reg3_path = r'C:\Data\PosLip\reg_3_ub'
# reg4_path = r'C:\Data\PosLip\reg_4_ub'
# reg5_path = r'C:\Data\PosLip\reg_5_ub'
#
# reg3_down = r'C:\Data\PosLip\reg_3\down_2'
#
# spectra_obj1 = loadmat(glob(os.path.join(regDir, '*reg_1.mat'))[0])
# spectra_obj2 = loadmat(glob(os.path.join(reg2_path, '*reg_2.mat'))[0])
# spectra_obj3 = loadmat(glob(os.path.join(reg3_path, '*reg_3.mat'))[0])
# spectra_obj4 = loadmat(glob(os.path.join(reg4_path, '*reg_4.mat'))[0])
# spectra_obj5 = loadmat(glob(os.path.join(reg5_path, '*reg_5.mat'))[0])
# # print(spectra_obj3.keys())
#
# spec_array1 = spectra_obj1['array']
# spec_data1 = spectra_obj1['spectra']
# spec_coor1 = spectra_obj1['coordinates']
#
# spec_array2 = spectra_obj2['array']
# spec_data2 = spectra_obj2['spectra']
# spec_coor2 = spectra_obj2['coordinates']
#
# spec_array3 = spectra_obj3['array']
# spec_data3 = spectra_obj3['spectra']
# spec_coor3 = spectra_obj3['coordinates']
#
# spec_array4 = spectra_obj4['array']
# spec_data4 = spectra_obj4['spectra']
# spec_coor4 = spectra_obj4['coordinates']
#
# spec_array5 = spectra_obj5['array']
# spec_data5 = spectra_obj5['spectra']
# spec_coor5 = spectra_obj5['coordinates']

# print(spec_coor1, '\n', spec_coor3)
# downArray, downSpec, downCoor = downSpatMS(spec_array3, 2, [3, 3, 1])

# print(spectra_obj1.keys())


# msmlfunc(mspath, regID=1, threshold=0.95, exprun='w_wvlt')
# msmlfunc2(posLip, spec_array, spec_data, spec_coor, 3, 0.95, 'array_ml')

# for down_i in [0, 1, 2, 3]:
#     print("downsample >>> ", down_i)


# msmlfunc2(posLip, downArray, downSpec, downCoor, 3, 0.95, 'down_ml')

# seg1_path = glob(os.path.join(reg1_path, '*binning2_gmm_6_3_1.npy'))[0]
# seg2_path = glob(os.path.join(reg2_path, '*binning2_gmm_6_3_1.npy'))[0] #*hdbscan-label.npy
# seg3_path = glob(os.path.join(reg3_path, '*binning2_gmm_6_3_1.npy'))[0]
# seg4_path = glob(os.path.join(reg4_path, '*binning2_gmm_6_3_1.npy'))[0]
# seg5_path = glob(os.path.join(reg5_path, '*binning2_gmm_6_3_1.npy'))[0]
#
# seg3_down = glob(os.path.join(reg3_down, '*6_3_1.npy'))[0]

# from Utilities import matchSpecLabel2
# matchSpecLabel2(True, seg1_path, seg2_path, arr1=spec_array1, arr2=spec_array2) #, arr5=spec_array5)

# +-------------------------+
# | check downsample 3x3    |
# +-------------------------+
if __name__ != '__main__':
    posLip = r'C:\Data\210427-Chen_poslip'  # r'C:\Data\PosLip'
    mspath = glob(os.path.join(posLip, '*.imzML'))[0]
    print(mspath)

    regID = 3

    # downArray, downSpec, downCoor = downSpatMS(spec_array3, 2, [3, 3, 1])
    for regID in [1, 2, 4, 5]:
        print("region >>", regID)
        for ds_i in tqdm(range(9), desc='downsamp%#'):
            print("downsample index >> ", ds_i)
            msmlfunc3(mspath, regID=regID, threshold=0.95, exprun='ds_3x3_umap_2k', downsamp_i=ds_i, wSize=[3, 3, 1])

if __name__ != '__main__':
    # for downsample #

    for regID in [1, 2, 4, 5]:
        # if regID == 2:
        #     break
        reg_path = r'C:\Data\210427-Chen_poslip\reg_{}'.format(regID)
        reg_down_0 = os.path.join(reg_path, 'down_0')
        reg_down_1 = os.path.join(reg_path, 'down_1')
        reg_down_2 = os.path.join(reg_path, 'down_2')
        reg_down_3 = os.path.join(reg_path, 'down_3')
        reg_down_4 = os.path.join(reg_path, 'down_4')
        reg_down_5 = os.path.join(reg_path, 'down_5')
        reg_down_6 = os.path.join(reg_path, 'down_6')
        reg_down_7 = os.path.join(reg_path, 'down_7')
        reg_down_8 = os.path.join(reg_path, 'down_8')

        # seg1_path = glob(os.path.join(reg1_path, '*label.npy'))[0]  #  '*binning2_gmm_6_3_1.npy'))[0]
        seg_down_0 = glob(os.path.join(reg_down_0, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]  #'*gmm_6_3_1.npy'))[0]
        # pint(seg3_down_0)                          ds_3x3_umap_2k_gmm_6_3_1_1
        seg_down_1 = glob(os.path.join(reg_down_1, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]  #'*gmm_6_3_1.npy'))[0]
        seg_down_2 = glob(os.path.join(reg_down_2, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]  #'*gmm_6_3_1.npy'))[0]
        seg_down_3 = glob(os.path.join(reg_down_3, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]  #'*gmm_6_3_1.npy'))[0]
        seg_down_4 = glob(os.path.join(reg_down_4, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]
        seg_down_5 = glob(os.path.join(reg_down_5, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]
        seg_down_6 = glob(os.path.join(reg_down_6, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]
        seg_down_7 = glob(os.path.join(reg_down_7, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]
        seg_down_8 = glob(os.path.join(reg_down_8, '*ds_3x3_umap_2k_gmm_6_3_1.npy'))[0]

        # spectra_obj1 = loadmat(glob(os.path.join(reg1_path, '*reg_1.mat'))[0])
        spec_obj_0 = loadmat(glob(os.path.join(reg_down_0, '*.mat'))[0])
        spec_obj_1 = loadmat(glob(os.path.join(reg_down_1, '*.mat'))[0])
        spec_obj_2 = loadmat(glob(os.path.join(reg_down_2, '*.mat'))[0])
        spec_obj_3 = loadmat(glob(os.path.join(reg_down_3, '*.mat'))[0])
        spec_obj_4 = loadmat(glob(os.path.join(reg_down_4, '*.mat'))[0])
        spec_obj_5 = loadmat(glob(os.path.join(reg_down_5, '*.mat'))[0])
        spec_obj_6 = loadmat(glob(os.path.join(reg_down_6, '*.mat'))[0])
        spec_obj_7 = loadmat(glob(os.path.join(reg_down_7, '*.mat'))[0])
        spec_obj_8 = loadmat(glob(os.path.join(reg_down_8, '*.mat'))[0])
        # spec_array1 = spectra_obj1['array']
        # spec_data1 = spectra_obj1['spectra']
        # spec_coor1 = spectra_obj1['coordinates']
        spec_array_0 = spec_obj_0['array']
        spec_array_1 = spec_obj_1['array']
        spec_array_2 = spec_obj_2['array']
        spec_array_3 = spec_obj_3['array']
        spec_array_4 = spec_obj_4['array']
        spec_array_5 = spec_obj_5['array']
        spec_array_6 = spec_obj_6['array']
        spec_array_7 = spec_obj_7['array']
        spec_array_8 = spec_obj_8['array']
        # spec_data1_0 = spec_obj1_0['spectra']
        # spec_coor1_0 = spec_obj1_0['coordinates']
        #
        # spec_array1_1 = spec_obj1_1['array']
        # spec_data1_1 = spec_obj1_1['spectra']
        # spec_coor1_1 = spec_obj1_1['coordinates']
        #
        # spec_array1_2 = spec_obj1_2['array']
        # spec_data1_2 = spec_obj1_2['spectra']
        # spec_coor1_2 = spec_obj1_2['coordinates']
        #
        # spec_array1_3 = spec_obj1_3['array']
        # spec_data1_3 = spec_obj1_3['spectra']
        # spec_coor1_3 = spec_obj1_3['coordinates']
        matchSpecLabel2(True,
                              seg_down_0,
                              seg_down_1,
                              seg_down_2,
                              seg_down_3,
                              seg_down_4,
                              seg_down_5,
                              # seg_down_6,
                              # seg_down_7,
                              # seg_down_8,

                                           arr1=spec_array_0,
                                           arr2=spec_array_1,
                                           arr3=spec_array_2,
                                           arr4=spec_array_3,
                                           arr5=spec_array_4,#)
                                           arr6=spec_array_5, exprun='region {}_umap_gmm'.format(regID))
                                           # arr7=spec_array_6,#)
                                           # arr8=spec_array_7,
                                           # arr9=spec_array_8)

if __name__ != '__main__':
    # for downsample #
    reg_path = r'C:\Data\210427-Chen_poslip\reg_3'
    seg_path = glob(os.path.join(reg_path, 'umap-umap+gmm_corr_2k_gmm_6_3_1.npy'))[0]
    spec_obj = loadmat(glob(os.path.join(reg_path, '*reg_3.mat'))[0])
    spec_array = spec_obj['array']
    matchSpecLabel2(True, seg_path, arr1=spec_array) #,

# +-------------------------+
# |  smoothing demo works   |
# +-------------------------+
if __name__ != '__main__':
    from Utilities import wavelet_denoising

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
    print(discreteWvList)
    # wvltList = pywt.wavelist()
    # print(len(wvltList))
    # print(wvltList[23])
    # print(pywt.families())
    # for family in pywt.families():
    #     print(family, ' : ', pywt.wavelist(family))
    nS = 1520 #np.random.randint(spectra.shape[0])
    signal = copy.deepcopy(spectra[nS, :])
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    # p = ax.plot(signal, 'g')
    p = ax.plot(signal, color=(0.9, 0, 0), linewidth=1.0, label='raw')
    # outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=wl_, polyorder=po_)
    filtered = wavelet_denoising(signal, wavelet=discreteWvList[2])  #'bior4.4')
    # p, = ax.plot(filtered)
    p, = ax.plot(filtered, color=(0, 0, 1), linewidth=1.0, label='filtered', alpha=0.5)
    plt.subplots_adjust(bottom=0.25)
    ax.set_xlabel("m/z(shifted)", fontsize=12)
    ax.set_ylabel("intensity", fontsize=12)
    ax.legend(loc='upper right')
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    wvlt = Slider(ax_slide, 'wavelet', valmin=0, valmax=len(discreteWvList)-1, valinit=0, valstep=1)
    print(wvlt.val)
    def update(val):
        current_wvlt = int(wvlt.val)
        print(discreteWvList[current_wvlt])
        filtered = wavelet_denoising(signal, wavelet='db8') #discreteWvList(current_wvlt))  # 'bior4.4')
        p.set_ydata(filtered)
        fig.suptitle('DWT #{} {}'.format(nS, discreteWvList(current_wvlt)), fontsize=12, y=1, fontweight='bold')
        fig.canvas.draw()

    # #     outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=current_v, polyorder=po_)
    wvlt.on_changed(update)
    plt.show()
# +-------------------+
# |   standardize     |
# +-------------------+
# +---------------------------+
# |   demonstrate smoothing   |
# +---------------------------+
if __name__ != '__main__':
    from Utilities import _smooth_spectrum
    nS = 1520   # np.random.randint(spectra.shape[0])
    signal = copy.deepcopy(spectra[nS, :])
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    p = ax.plot(signal, color=(0.9, 0, 0), linewidth=1.0, label='raw')
    wl_ = 5
    po_ = 2
    outspectrum = _smooth_spectrum(signal, method='savgol', window_length=wl_, polyorder=po_)
    p, = ax.plot(outspectrum, color=(0, 0, 1), linewidth=1.0, label='filtered', alpha=0.5)
    plt.subplots_adjust(bottom=0.25)
    ax.set_xlabel("m/z(shifted)", fontsize=12)
    ax.set_ylabel("intensity", fontsize=12)
    ax.legend(loc='upper right')
    fig.suptitle('Savitzky-Golay filtering', fontsize=12, y=1, fontweight='bold')
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    win_len = Slider(ax_slide, 'window length', valmin=wl_, valmax=99, valinit=99, valstep=2)
    def update(val):
        current_v = int(win_len.val)
        #     try:
        #         filtered = wavelet_denoising(signal, wavelet=wav, level=1)
        #     except:
        #         pass
        outspectrum = _smooth_spectrum(signal, method='savgol', window_length=current_v, polyorder=po_)
        p.set_ydata(outspectrum)
        fig.canvas.draw()
    win_len.on_changed(update)
    plt.show()
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
# coif4, coif8, coif9, coif10, coif11, coif12
def create_img(nX, nY, nMZ):
    img = np.zeros((nX,nY,nMZ))
    img_sp = np.arange(nX*nY).reshape(nX, nY) + 1
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            img[r,c,:] = img_sp[r,c]
    # print(img.shape)
    return img
# image = create_img(7, 5, 10)

# # print("downsampled", image_.shape)
#
# fig, ax = plt.subplots(1, 2, figsize=(20, 8), dpi=200)
# ax[0].imshow(spectra_array[..., 10])
# ax[1].imshow(downArray[..., 10])
# plt.show()

# print(downSpec.shape, len(downCoor))
# print(downCoor)

# msmlfunc2(posLip, downArray, downSpec, downCoor, 3, 0.95, 'down_ml')
# array = spectra_array_['array']
# spectra = spectra_array_['data']
# coordinates = spectra_array_['coordinates']









