import os
import copy
from glob import glob
import numpy as np
import pywt
from Utilities import msmlfunc3, downSpatMS, matchSpecLabel2
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib as mtl
mtl.use('TkAgg')    # required for widget slider...
from scipy.io import loadmat, savemat
import time
from imzml import IMZMLExtract, normalize_spectrum

posLip = r'C:\Data\210427-Chen_poslip' #r'C:\Data\PosLip'
mspath = glob(os.path.join(posLip, '*.imzML'))[0]
print(mspath)

# ImzObj = IMZMLExtract(mspath)
regID = 5

msmlfunc3(mspath, regID=regID, threshold=0.95, exprun='umap+gmm_corr_2k', downsamp_i=None, wSize=None)

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
# matchSpecLabel2(True, seg1_path, seg2_path, seg3_path, seg4_path, arr1=spec_array1, arr2=spec_array2, arr3=spec_array3, arr4=spec_array4) #, arr5=spec_array5)

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

if __name__ != '__main__':
    # wvltList = pywt.wavelist()
    # print(len(wvltList))
    # print(wvltList[23])
    # print(pywt.families())
    # for family in pywt.families():
    #     print(family, ' : ', pywt.wavelist(family))

    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    p = ax.plot(signal)
    # outspectrum = _smooth_spectrum(refSpec, method='savgol', window_length=wl_, polyorder=po_)
    filtered = wavelet_denoising(signal, wavelet=discreteWvList[2]) #'bior4.4')
    p, = ax.plot(filtered)
    plt.subplots_adjust(bottom=0.25)
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    wvlt = Slider(ax_slide, 'wavelet', valmin=0, valmax=len(discreteWvList)-1, valinit=0, valstep=1)
    print(wvlt.val)
    def update(val):
        current_wvlt = int(wvlt.val)
        print(discreteWvList[current_wvlt])
        filtered = wavelet_denoising(signal, wavelet=discreteWvList(current_wvlt))  # 'bior4.4')
        p.set_ydata(filtered)
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
    nS = np.random.randint(spectra_array.shape[0])
    signal = copy.deepcopy(spectra_array[nS, :])
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    p = ax.plot(signal, color=(0.9, 0, 0), linewidth=1.0, label='raw')
    wl_ = 3  # todo: 5 is better
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









