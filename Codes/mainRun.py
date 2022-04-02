import os
from glob import glob
import numpy as np
from scipy.io import savemat, loadmat
from matplotlib.widgets import Slider
from Utilities import Binning2
from imzml import IMZMLExtract, normSpec, normalize_spectrum
import seaborn as sns
import pywt
import pandas as pd
import time
import copy
from IPython import get_ipython
import matplotlib as mtl
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



# posLip = r'/media/banikr2/DATA/MALDI/210427_Chen_pos_lipid' #
posLip = r'C:\Data\PosLip'
mspath = glob(os.path.join(posLip, '*.imzML'))[0]
print(mspath)

# msmlfunc(mspath, regID=1, threshold=0.95, exprun_name='sav_golay_norm')   # todo: change values

if __name__ != '__main__':
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
if __name__ == '__main__':
    def madev(d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def wavelet_denoising(x, wavelet='bior4.4', level=2):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1/0.6745) * madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')

    nX = 90
    nY = 41
    signal = spectra0_orig[nX, nY, :][200:400]

    # for wav in pywt.wavelist():
    #     print(wav)
    print(pywt.wavelist())
    # todo: run all wavelet simulation ...

    #     try:
    #         filtered = wavelet_denoising(signal, wavelet=wav, level=1)
    #     except:
    #         pass

    filtered = wavelet_denoising(signal, wavelet='bior4.4')
    # plt.figure(figsize=(10, 6))
    # plt.plot(signal, label='Raw')
    # plt.plot(filtered, label='Filtered')
    # plt.legend()
    # plt.title(f"DWT Denoising with Wavelet", size=15)
    # plt.show()
    #
    # fig, ax = plt.subplots(2, 1, figsize=(20, 8), dpi=200)
    # # plt.subplot(411)
    # ax[0].plot(signal)
    # plt.title("raw")
    # # plt.subplot(412)
    # ax[1].plot(filtered)
    # plt.title("filtered")
    # plt.show()
    # print(np.min(filtered))

if __name__ != '__main__':
    wavelet = 'bior4.4'
    nX = 90  # np.random.randint(spectra0_orig.shape[0])
    nY = 41  # np.random.randint(spectra0_orig.shape[1])
    # print(nX, nY)

    data_wt = pywt.dwt(refSpec, wavelet, mode='symmetric', axis=-1)
    # plt.plot(data_wt)
    # plt.show()
    print(np.array(data_wt).shape)

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
