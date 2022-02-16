import os
from glob import glob
from pIMZ.imzml import IMZMLExtract
from pIMZ.comparative import CombinedSpectra
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# imze.list_regions()
# print(len(imze.mzValues))   # 29888
# print(min(imze.mzValues), max(imze.mzValues))

# spectra1_orig = imze.get_region_array(1, makeNullLine=True)
# spectra1_intra = imze.normalize_region_array(spectra1_orig, normalize="intra_median")
# spectra1 = imze.normalize_region_array(spectra1_intra, normalize="inter_median")

# imze.plot_fcs(spectra1_orig, [(5, 30), (10, 30), (20, 30), (25, 30), (35, 30), (40, 30)])

# imze.list_highest_peaks(spectra1, True)

# plt.plot()
# reg_spec = imze.get_region_spectra(1)
# print(type(reg_spec))
# plt.show()
# print(reg_spec)

# +------------------------+
# | why mz values differ ? |
# +------------------------+
if __name__ != '__main__':
    mzlist = imze.parser.getspectrum(1)[0]
    print(len(mzlist))
    (minx, maxx), (miny, maxy), (minz, maxz), spectraLength, longest_mz_length_in_reg, spec_w_longest_mz_in_reg = imze.get_region_range(1)
    print(longest_mz_length_in_reg, spec_w_longest_mz_in_reg)
    region1_longest_mz = imze.parser.getspectrum(spec_w_longest_mz_in_reg)
    print(np.shape(region1_longest_mz))
    plt.plot(imze.parser.getspectrum(spec_w_longest_mz_in_reg)[0],
             imze.parser.getspectrum(spec_w_longest_mz_in_reg)[1])
    plt.show()

if __name__ != '__main__':
    # avg_reg_spec = imze.get_avg_region_spectrum(1)
    region1 = imze.get_region_array(1)
    print("Here", np.shape(region1))  # (36755,)
    masses = imze.parser.getspectrum(10)[0]
    print(np.shape(masses))
    peakarray, peakmasses = imze.to_called_peaks(region1, masses, resolution=4)
    peakimg = np.mean(peakarray, axis=2)
    print(peakimg.shape, "<<< Image shape")
    plt.imshow(peakimg)
    plt.colorbar()
    plt.show()
    print("Ends")

# +------------------------+
# | get_region_index_array |
# +------------------------+
if __name__ != '__main__':
    reg_ = imze.get_region_index_array(1)
    print("specID range", reg_.max(), reg_.min())
    print(reg_.shape)
    plt.imshow(reg_)
    plt.colorbar()
    plt.show()

# def reg_to_tot_coord(imze, regionID):

# +------------------+
# | get_region_range |
# +------------------+
if __name__ != '__main__':
    xr, yr, zr, sc, _, _ = imze.get_region_range(regionid)
    print(xr, yr, zr, sc)

# +--------------------------+
# | coord to index in region |
# +--------------------------+
if __name__ != '__main__':
    rs = imze.get_region_shape(regionid)
    # sarray = np.zeros((rs[0], rs[1]), dtype=np.float32)
    spectrareg1 = imze.get_region_indices(regionid)
    print(spectrareg1)
    spectraIDList1 = []
    for coord in spectrareg1:
        spectraIDList1.append(spectrareg1[coord])
    print(spectraIDList1)
    specarray = imze.parser.getspectrum(spectraIDList1[0])
    # +-----------------------+
    # | compare two m/z  s    |
    # +-----------------------+
    if __name__ != '__main__':
        mz0 = imze.parser.getspectrum(spectraIDList1[0])[0]
        print(">>", len(mz0))
        mz1 = imze.parser.getspectrum(spectraIDList1[12])[0]
        plt.subplot(121)
        plt.plot(mz0, imze.parser.getspectrum(spectraIDList1[0])[1])
        plt.title("m/z: 0")
        plt.subplot(122)
        plt.plot(mz1, imze.parser.getspectrum(spectraIDList1[12])[1])
        plt.show()
        # print(len(mz1) - len(mz0))
        # mz0_ = np.pad(mz0, (0, len(mz1) - len(mz0)), mode='constant', constant_values=0)
        count = 0
        for m0 in mz0:
            if (mz1 == m0).sum() > 0:
                count += 1
        print("***", count)


    # print(np.shape(specarray))
    # print(specarray)
    # plt.plot(specarray[0], specarray[1])
    # plt.show()
    # region1 = imze.get_region_array(1)
    # xpos = coord[0] - xr[0]
    # ypos = coord[1] - yr[0]
    # # print(reg_[xpos, ypos].shape)
    # plt.plot(region1[xpos, ypos, :])
    # plt.title("From region array")
    # plt.show()
    # {(545, 262, 1): 1182, (545, 263, 1): 1266, (545, 264, 1): 1350, (545, 265, 1): 1434, ...
    if __name__ != '__main__':
        for coord in spectrareg1:
            xpos = coord[0] - xr[0]
            ypos = coord[1] - yr[0]
            specIdx = spectrareg1[coord]
            sarray[xpos, ypos] = np.mean(imze.parser.getspectrum(specIdx))
        plt.imshow(sarray)
        plt.colorbar()
        plt.show()

# +---------------+
# | applying PCA  |
# +---------------+
if __name__ != '__main__':
    print(len(imze.parser.getspectrum(10)[0]))  # 36755
    masses = imze.parser.getspectrum(10)[0]
    print(min(masses), max(masses))
    massesNew = [x for x in np.arange(min(masses), max(masses), 0.1)]
    print(len(massesNew))  # 14932
    ijspec = imze.parser.getspectrum(10)[1]
    # arr1_interp = interpolate.interp1d(masses, ijspec)
    # specNew = arr1_interp(massesNew)

    # f = interpolate.Akima1DInterpolator(masses, ijspec)
    # specNew = f(massesNew)
    f = interpolate.CubicSpline(masses, ijspec)
    specNew = f(massesNew)
    plt.subplot(211)
    plt.plot(masses[10:], ijspec[10:])
    plt.subplot(212)
    plt.plot(massesNew[10:], specNew[10:])
    plt.show()
    #
    # arr_ref = np.array([1, 5, 2, 3, 7, 1])  # shape (6,), reference
    # arr1 = np.array([1, 5, 2, 3, 7, 2, 1])  # shape (7,), to "compress"
    # arr2 = np.array([1, 5, 2, 7, 1])  # shape (5,), to "stretch"
    # print(np.arange(arr1.size))

    # arr1_interp = interpolate.interp1d(np.arange(arr1.size), arr1)
    # arr1_compress = arr1_interp(np.linspace(0, arr1.size - 1, arr_ref.size))
    # arr2_interp = interpolate.interp1d(np.arange(arr2.size), arr2)
    # arr2_stretch = arr2_interp(np.linspace(0, arr2.size - 1, arr_ref.size))

if __name__ != '__main__':
    sarray = imze.get_region_array(1)
    print(sarray.shape)
    sarray_ = sarray #[42:46, 24:28, :]
    # print("sarray -> ", sarray_.shape, type(sarray_))
    # sarray_ = np.array(sarray_)
    sarray_ = sarray_.transpose(2, 0, 1).reshape(sarray_.shape[2], -1)
    # sarray_ = sarray_.flatten()
    # sarray_ = np.reshape(sarray, (-1, 1))
    print("sarray -> ", sarray_.shape, type(sarray_))
    # sarray_flat = np.array(sarray_).reshape(8, sarray_.shape[2])
    # print("what: ", np.shape(sarray_flat))
    # sarray_flat = np.zeros(8, sarray_.shape[2])
    # sarray_flat[0:4, :] = sarray_[:, ]
    # sarray_ = sarray_.transpose(1, 0)
    print(np.shape(sarray_))

if __name__ != '__main__':
    from sklearn.decomposition import PCA
    import mglearn
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    pca.fit(sarray_)
    x_pca = pca.transform(sarray_)
    print(x_pca.shape)
    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(x_pca[:, 0], x_pca[:, 1])
    plt.show()

    from sklearn.manifold import TSNE

    tsne = TSNE(random_state=17)

    X_tsne = tsne.fit_transform(sarray_)

    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.colorbar()
    plt.title('t-SNE projection')
    plt.show()

if __name__ != '__main__':
    print(len(sarray_[1111, :]))
    print(np.unique(sarray_[1111, :]))

if __name__ != '__main__':
    class Binning(object):
        """
        given the imze object should create 3D matrix(spatial based) or 2D(spectrum based)
        spectrum: array with 2 vectors, one of abundance(1), other with m/z values(0)
        n_bins: number of bins/samples to be digitized
        plotspec: to plot the new binned spectrum, default--> True
        """

        def __init__(self, imzObj, regionID, n_bins, plotspec=False):
            self.imzObj = imzObj
            self.regionID = regionID
            self.n_bins = n_bins
            self.plotspec = plotspec

            xr, yr, zr, _ = self.imzObj.get_region_range(regionID)
            self.imzeShape = [xr[1] - xr[0] + 1,
                              yr[1] - yr[0] + 1, self.n_bins - 1]

        def getBinMat(self):
            sarray = np.zeros(self.imzeShape)
            regInd = self.imzObj.get_region_indices(self.regionID)
            binned_mat = np.zeros([len(regInd), self.n_bins - 1])
            coordList = []
            for i, coord in enumerate(regInd):
                spectrum = self.imzObj.parser.getspectrum(self.imzObj.coord2index.get(coord))  # [0]
                bSpec = self.onebinning(spectrum)
                binned_mat[i] = bSpec
                xpos = coord[0] - xr[0]
                ypos = coord[1] - yr[0]
                sarray[xpos, ypos, :] = bSpec
                coordList.append(coord)
            return sarray, binned_mat, coordList

        #     def getRegMat(self):

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

    A = Binning(imzObj=imze, region=1, n_bins=1495)

    mat, _ = A.getBinMat()

    print(mat.shape)

if __name__ != '__main__':
    from sklearn.decomposition import PCA
    import mglearn
    import matplotlib.pyplot as plt

    pca = PCA(n_components=10)
    pca.fit(mat)
    x_pca = pca.transform(mat)
    print(x_pca.shape)
    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(x_pca[:, 0], x_pca[:, 9])
    plt.show()

if __name__ != '__main__':
    mat = x_pca
    import seaborn as sns
    import pandas as pd

    data = pd.DataFrame(mat)
    corr = data.corr()
    ax = sns.heatmap(corr,               #mat,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(0, 220, n=200),
            square=True)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right')

    plt.title("Correlation matrix heatmap")
    plt.show()

if __name__ == '__main__':
    from pIMZ.regions import SpectraRegion
    poslipPath = r'/media/banikr2/DATA/MALDI/fromCardinal/PosLip'
    plipImzFile = glob(os.path.join(poslipPath, '*.imzML'))
    print(plipImzFile)
    imze = IMZMLExtract(plipImzFile[0])
    regionID = 1
    spectra_orig = imze.get_region_array(regionID, makeNullLine=True)
    # print("This: ", spectra_orig.shape)
    spectra_intra = imze.normalize_region_array(spectra_orig, normalize="intra_median")
    spectra = imze.normalize_region_array(spectra_intra, normalize="inter_median")
    spec = SpectraRegion(spectra, imze.mzValues)
    spec.calculate_similarity(mode="spectra_log_dist")
    spec.segment(method="WARD", number_of_regions=5)
    spec.plot_segments()