import os
import copy
from glob import glob
import math
import numpy as np
import pywt
from Utilities import msmlfunc4, matchSpecLabel2, ImzmlAll, rawVSprocessed
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib as mtl
# mtl.use('TkAgg')    # required for widget slider...
from scipy.io import loadmat, savemat
import time
from imzml import IMZMLExtract, normalize_spectrum
from scipy import interpolate
from pyimzml.ImzMLParser import ImzMLParser
from ms_peak_picker import pick_peaks
import h5py

# posLip = r'C:\Data\210427-Chen_poslip' #r'C:\Data\PosLip'
posLip = r'/media/banikr/DATA/MALDI/demo_banikr_'
mspath = glob(os.path.join(posLip, '*.imzML'))[0]
print(mspath)
# ImzObj = ImzmlAll(mspath)
# (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = ImzObj.get_region_range(2)
# print((minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx)
# array3D, array2D, longestmz, regionshape, lCoorIdx=ImzObj.get_region(regID, whole=True)
# print(array2D.shape)
# spectra, peak_mz = ImzObj.peak_pick(, refmz2)

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

if __name__ != '__main__':
    regID = 5
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
    regID = 5
    spec3D5, spectra5, refmz5, regionshape5, localCoor5 = ImzObj.get_region(regID, whole=True)

    print("3d reg 1", spec3D1.shape)
    print("2d reg 1 >>", spectra1.shape)

    print("3d reg 2", spec3D2.shape)
    print("2d reg 2 >>", spectra2.shape)
    print("3d reg 3", spec3D3.shape)
    print("2d reg 3 >>", spectra3.shape)
    print("3d reg 5", spec3D5.shape)
    print("2d reg 5 >>", spectra5.shape)

    seg1 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_1', '*4_1.npy'))[0]
    seg2 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_2', '*4_1.npy'))[0]
    seg3 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_3', '*4_1.npy'))[0]
    seg5 = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_5', '*4_1.npy'))[0]

    label1 = np.load(seg1)
    label2 = np.load(seg2)
    label3 = np.load(seg3)
    label5 = np.load(seg5)
    # print(np.unique(label2))
    # print(np.unique(label3))
    elements, counts = np.unique(label2, return_counts=True)
    print("label2 >>", elements, counts)
    elements, counts = np.unique(label3, return_counts=True)
    print("label3 >>", elements, counts)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    arrays = [label1, label2, label3, label5]
    title = ['reg1', 'reg2', 'reg3', 'reg5']
    butterfly_labels = [1, 2, 1, 2] # butterfly
    peripheral_labels = [3, 3, 2, 1]  # peripheral
    gm_labels = [2, 1, 4, 3]
    rest = [4, 4, 3, 4]

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax in zip(arrays, title, axs.ravel()):
        im = ax.imshow(ar)  # , cmap='twilight') #cm)
        ax.set_title(tl, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=ax)
    plt.show()

    fig, axs = plt.subplots(1, len(title), figsize=(10, 8), dpi=200, sharex=False)
    for ar, tl, ax, bl in zip(arrays, title, axs.ravel(), butterfly_labels):
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
        print(_labels)
        _labels = list(_labels)
        spec_lab_1 = []
        spec_lab_2 = []  # np.zeros([len(np.where(label1 == 4)[0]), spectra.shape[1]], dtype=np.float32)
        spec_lab_3 = []  # np.zeros([len(np.where(label1 == 3)[0]), spectra.shape[1]], dtype=np.float32)
        spec_lab_5 = []
        img = np.zeros([regionshape1[0], regionshape1[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label1 == _labels[0])[0], np.where(label1 == _labels[0])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor1):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_lab_1.append(spectra1[ldx, :])  # [idx, :] = spectra[ldx, :]

        img = np.zeros([regionshape2[0], regionshape2[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label2 == _labels[1])[0], np.where(label2 == _labels[1])[1])):
            # print(idx, i, j, np.where(localCoor[:] == (i, j)))
            for ldx, item in enumerate(localCoor2):
                if item == (i, j):
                    # print(idx, i, j, ldx)
                    img[i, j] = 2
                    spec_lab_2.append(spectra2[ldx, :])  #[idx, :] = spectra[ldx, :]
                    # break
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
                    # break
        plt.imshow(img)
        plt.show()
        img = np.zeros([regionshape5[0], regionshape5[1]], dtype=np.float32)
        for idx, (i, j) in enumerate(zip(np.where(label5 == _labels[3])[0], np.where(label5 == _labels[3])[1])):
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
        spec_lab_5 = np.array(spec_lab_5)
        print("b4 1-way >>", spec_lab_2.shape, spec_lab_5.shape)
        from scipy.stats import stats
        # fvalue, pvalue = stats.f_oneway(np.mean(spec_lab_2, axis=0), np.mean(spec_lab_3, axis=0), axis=0)
        # fvalue, pvalue = stats.f_oneway(spec_lab_3[0:1000, :], spec_lab_3[1000:, :]) #, axis=0)
        Fvalue, pvalue = stats.f_oneway(spec_lab_1, 0.999*spec_lab_1, 0.9987* spec_lab_1, 1.002*spec_lab_1)  # , axis=0)
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
        break
    #     pvalue_log = -np.log10(pvalue)
    #     logpvalues.append(pvalue_log)
    #     plt.plot(pvalue_log)
    #     plt.title("-log10(p-value)")
    #     plt.show()
    # dANOVA = {'Fvalue':Fvalues,
    #           'pvalue':pvalues,
    #           'logpvalue':logpvalues
    #          }
    # aPath = os.path.join(os.path.dirname(mspath), 'ANOVA_results.bin')
    # # print(mzPath)
    # with open(aPath, 'wb') as pfile:
    #     pickle.dump(dANOVA, pfile)
    #
    # import seaborn as sns
    # import pandas as pd
    # p_values = pd.DataFrame(pvalues)
    # ax = sns.boxplot(data=p_values.T)
    # plt.show()
    # Fvalues = pd.DataFrame(Fvalues)
    # ax = sns.boxplot(data=Fvalues.T)
    # plt.show()
    # logp_values = pd.DataFrame(logpvalues)
    # ax = sns.boxplot(data=logp_values.T)
    # plt.title("-log10(p-value)")
    # plt.show()

# +--------------------+
# |    ANOVA results   |
# +--------------------+
if __name__ == '__main__':
    aPath = os.path.join(os.path.dirname(mspath), 'ANOVA_results.bin')
    with open(aPath, 'rb') as pfile:
        dANOVA = pickle.load(pfile)
    print(dANOVA.keys())
    Fv = dANOVA['Fvalue']
    pv = dANOVA['pvalue']
    # print(np.isnan(pv))
    pv = np.nan_to_num(pv, nan=0.98) #random.uniform(0.95, 1))
    # print(np.isnan(pv))
    logpv = -np.log10(pv) #dANOVA['logpvalue']
    # print(np.shape(pv))  #(4, 36996)
    # print(logpv)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)
    labels = ['butterfly', 'peripheral', 'gm', 'rest']
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
    bplot = ax1.boxplot(list(Fv),
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
    ax1.set_title('ANOVA: p-value', fontsize=16)
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

if __name__ != '__main__':
    seg1_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_1', '*4_1.npy'))[0]
    seg2_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_2', '*4_1.npy'))[0]
    seg3_path = glob(os.path.join(r'/media/banikr/DATA/MALDI/demo_banikr_/reg_3', '*4_1.npy'))[0]
    ImzObj = ImzmlAll(mspath)
    regID = 1
    spec3D1, spectra1, refmz1, regionshape1, localCoor1 = ImzObj.get_region(regID)
    regID = 2
    spec3D2, spectra2, refmz2, regionshape2, localCoor2 = ImzObj.get_region(regID)
    regID = 3
    spec3D3, spectra3, refmz3, regionshape3, localCoor3 = ImzObj.get_region(regID)
    matchSpecLabel2(True, seg1_path, seg2_path, seg3_path, arr1=spec3D1, arr2=spec3D2, arr3=spec3D3, exprun='kinda whole')

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









