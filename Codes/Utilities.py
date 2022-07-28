import os
import time
import math
import numpy as np
import numba as nb
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import copy
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from scipy.sparse import diags, spdiags, linalg
from scipy.io import loadmat, savemat
from scipy import ndimage, signal, interpolate, stats
from scipy.spatial.distance import cosine
import scipy.cluster.hierarchy as sch
from scipy.linalg import svd
import pywt
import mglearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP
import hdbscan
import matplotlib as mtl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns
from pyimzml.ImzMLParser import ImzMLParser, _bisect_spectrum
import ms_peak_picker
# from imzml import IMZMLExtract, normalize_spectrum
# from matchms import Spectrum, calculate_scores
# from matchms.similarity import CosineGreedy, CosineHungarian, ModifiedCosine
from tqdm import tqdm, tqdm_notebook
import joblib
from collections import defaultdict
import cv2

def _2d_to_3d(array2d, regionshape, Coord):
    nPixels, nMz = array2d.shape
    array3d = np.zeros([regionshape[0], regionshape[1], nMz])
    for idx, c in enumerate(Coord):
        array3d[c[0], c[1], :] = array2d[idx, :]
    return array3d

def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=105, porder=2, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    example: smoothspec = airPLS(spec)

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user.
                 The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    z = x - z
    return z

def findValInd(_list, func=max):
    """
    :param _list: the list
    :param func: max default, min or so
    :return: value and index
    """
    _value = func(_list)
    _index = _list.index(_value)
    return _value, _index

class ImzmlAll(object):
    def __init__(self, mspath):
        self.mspath = mspath
        self.parser = ImzMLParser(mspath)   # this is ImzObj

    def _find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")  # Find indices
        if idx > 0 and (
                idx == len(array)
                or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
        ):
            return idx - 1
        else:
            return idx

    def baseline_als_optimized(self, spec, lam=105, p=1e-3, niter=10):
        L = len(spec)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = linalg.spsolve(Z, w * spec)
            w = p * (spec > z) + (1 - p) * (spec < z)
        #     print(w)
        #     z[z<0]=0
        z_ = copy.deepcopy(z)
        z_[z_ > 0] = 0
        z += abs(z_)
        return z

    def _interpolate_spectrum(self, spec, masses, masses_new, method="Pchip"):
        """
        Args:
            spec (list/numpy.array, optional): spectrum
            masses (list): list of corresponding m/z values (same length as spectra)
            masses_new (list): list of m/z values
            method (str, optional):  Method to use to interpolate the spectra: "akima",
            "interp1d", "CubicSpline", "Pchip" or "Barycentric". Defaults to "Pchip".

        Returns:
            lisr: updatedd spectrum
        """
        if method == "akima":
            f = interpolate.Akima1DInterpolator(masses, spec)
            specNew = f(masses_new)
        elif method == "interp1d":
            f = interpolate.interp1d(masses, spec)
            specNew = f(masses_new)
        elif method == "CubicSpline":
            f = interpolate.CubicSpline(masses, spec)
            specNew = f(masses_new)
        elif method == "Pchip":
            f = interpolate.PchipInterpolator(masses, spec)
            specNew = f(masses_new)
        elif method == "Barycentric":
            f = interpolate.BarycentricInterpolator(masses, spec)
            specNew = f(masses_new)
        else:
            raise Exception("Unknown interpolation method")
        return specNew

    def _global2index(self):
        """
        global coordinates: index
        """
        global2index = {}
        for sidx, coord in enumerate(self.parser.coordinates):
            global2index[coord] = sidx
        return global2index

    def _get_regions(self):
        maxX = self.parser.imzmldict['max count of pixels x']
        maxY = self.parser.imzmldict['max count of pixels y']
        img = np.zeros((maxX + 1, maxY + 1))
        for coord in self.parser.coordinates:
            img[coord[0], coord[1]] = 1
        labeled_array, num_features = ndimage.label(img, structure=np.ones((3, 3)))
        # print("There are {} regions.".format(num_features))
        return labeled_array, num_features

    def get_region_pixels(self, regID=None):
        """
        :param regID:
        :return: global coordinates
        """
        regionCoords = defaultdict(list)
        labeled_array, num_features = self._get_regions()
        # plt.imshow(labeled_array)
        # plt.show()
        for x in range(0, labeled_array.shape[0]):
            for y in range(0, labeled_array.shape[1]):
                if labeled_array[x, y] == 0:
                    continue
                regionCoords[labeled_array[x, y]].append((x, y, 1))

        if regID is not None:
            regionPixels = regionCoords[regID]  # [(704, 180, 1), (705, 178, 1), ...
            return regionPixels
        else:
            return regionCoords

    def get_region_shape_coords(self, regID):
        gcoord2index = self._global2index()
        spectralength = 0
        mzidx = 0
        regionPixels = self.get_region_pixels(regID)
        minx = min([x[0] for x in regionPixels])
        maxx = max([x[0] for x in regionPixels])
        miny = min([x[1] for x in regionPixels])
        maxy = max([x[1] for x in regionPixels])
        minz = min([x[2] for x in regionPixels])
        maxz = max([x[2] for x in regionPixels])
        regionshape = [maxx - minx + 1,
                       maxy - miny + 1]
        if maxz - minz + 1 > 1:
            regionshape.append(maxz - minz + 1)

        lCoorIdx = []
        for idx, coord in enumerate(regionPixels):
            xpos = coord[0] - minx
            ypos = coord[1] - miny
            lCoorIdx.append((xpos, ypos))
        return regionshape, lCoorIdx

    def get_region_range(self, regID, whole=False):
        """
        regID: if None takes all regions
        """
        gcoord2index = self._global2index()
        spectralength = 0
        mzidx = 0
        regionPixels = self.get_region_pixels(regID)
        minx = min([x[0] for x in regionPixels])
        maxx = max([x[0] for x in regionPixels])
        miny = min([x[1] for x in regionPixels])
        maxy = max([x[1] for x in regionPixels])
        minz = min([x[2] for x in regionPixels])
        maxz = max([x[2] for x in regionPixels])
        if not whole: #regID is not None:
            # regionPixels = self.get_region_pixels(regID)
            # minx = min([x[0] for x in regionPixels])
            # maxx = max([x[0] for x in regionPixels])
            # miny = min([x[1] for x in regionPixels])
            # maxy = max([x[1] for x in regionPixels])
            # minz = min([x[2] for x in regionPixels])
            # maxz = max([x[2] for x in regionPixels])
            for coord in regionPixels:
                if self.parser.mzLengths[gcoord2index[coord]] > spectralength:
                    mzidx = gcoord2index[coord]
                    spectralength = self.parser.mzLengths[gcoord2index[coord]]
                # spectralength = max(spectralength,
                #                       self.parser.mzLengths[gcoord2index[coord]])
            return (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx
        else: # considers mz of all regions
            # regionCoords = self.get_region_pixels()
            # minx = np.inf
            # maxx = -np.inf
            # miny = np.inf
            # maxy = -np.inf
            # minz = np.inf
            # maxz = -np.inf
            # for i in range(len(regionCoords)):
            #     minx = min(minx, min([x[0] for x in regionCoords[i + 1]]))
            #     miny = min(miny, min([x[1] for x in regionCoords[i + 1]]))
            #     minz = min(minz, min([x[2] for x in regionCoords[i + 1]]))
            #     maxx = max(maxx, max([x[0] for x in regionCoords[i + 1]]))
            #     maxy = max(maxy, max([x[1] for x in regionCoords[i + 1]]))
            #     maxz = max(maxz, max([x[2] for x in regionCoords[i + 1]]))
            for sidx, coor in enumerate(self.parser.coordinates):
                # print(sidx, coor)
                if self.parser.mzLengths[sidx] > spectralength:
                    mzidx = sidx
                    spectralength = self.parser.mzLengths[mzidx]
            return (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx

    def get_region_shape(self, regID):
        (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = self.get_region_range(regID)
        regionshape = [maxx - minx + 1,
                       maxy - miny + 1]
        if maxz - minz + 1 > 1:
            regionshape.append(maxz - minz + 1)
        return regionshape

    def get_region_spectra(self, regID):
        labeled_array, num_features = self._get_regions()
        spectra = []
        for i, (x, y, z) in enumerate(tqdm(self.parser.coordinates, desc='listing spectra...')):
            if labeled_array[x, y] == regID:
                mz, ints = self.parser.getspectrum(i)
                spectra.append([mz, ints])
        return spectra

    def get_region_data(self, regID, whole=False):
        "data structure/matrix of the region"
        # if whole:
        #     (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = self.get_region_range()
        # else:
        (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = self.get_region_range(regID, whole)
        regionshape = [maxx-minx+1,
                       maxy-miny+1]
        if maxz-minz+1 > 1:
            regionshape.append(maxz-minz+1)
        regionshape.append(spectralength)
        gcoord2index = self._global2index()
        # array3D = np.zeros(regionshape, dtype=np.float32)
        regionPixels = self.get_region_pixels(regID)
        array2D = np.zeros([len(regionPixels), spectralength], dtype=np.float32)
        longestmz = self.parser.getspectrum(mzidx)[0]
        # regCoor = np.zeros([len(regionPixels), 2])
        lCoorIdx = []   #defaultdict(list)
        nS = np.random.randint(len(regionPixels))
        for idx, coord in enumerate(regionPixels):
            xpos = coord[0] - minx
            ypos = coord[1] - miny
            # regCoor[idx, 0], regCoor[idx, 1] = xpos, ypos
            spectra = self.parser.getspectrum(gcoord2index[coord])
            interp_spectra = self._interpolate_spectrum(spectra[1], spectra[0], longestmz, method='Pchip')
            if idx == nS:
                print("Plotting interpolation: #{}".format(nS))
                rawVSprocessed(spectra[0], spectra[1], longestmz, interp_spectra, n_spec=nS, exprun='Interpolation')
            # array3D[xpos, ypos, :] = interp_spectra
            array2D[idx, :] = interp_spectra
            # lCoorIdx[idx].append(gcoord2index[coord]) # {0: [1182], 1: [1266], 2: [1350], 3: [1434]
            lCoorIdx.append((xpos, ypos))
        return array2D, longestmz, regionshape, lCoorIdx # array3D, regionPixels

    # @np.vectorize
    def resample_region(self, regID, tol=0.02, savedata=True):
        """
        resamples spectra with equal bin width. Considers m/z range for all sections/regions
        """
        # regname = os.path.join(), 'reg_{}_tol_{}.h5'.format(regID, tol))
        # print(os.path.isfile(regname))
        regDir = os.path.join(os.path.dirname(self.mspath), 'reg_{}'.format(regID))
        if not os.path.isdir(regDir):
            os.mkdir(regDir)
        resampfilename = os.path.join(regDir, 'resampled_reg_{}_tol_{}.h5'.format(regID, tol))
        if os.path.isfile(resampfilename):
            print("Previous upsampling found. Fetching...")
            with h5py.File(resampfilename, 'r') as pfile:
                array2D = np.array(pfile.get('spectra'))
                massrange = np.array(pfile.get('mzrange'))
                regionshape = np.array(pfile.get('regionshape'))
                lCoorIdx = list(pfile.get('coordinates'))
        else:
            print("No previous upsampling found. Performing...")
            minmz = np.inf
            maxmz = -np.inf
            for s in range(len(self.parser.mzLengths)):
                minmz = min(self.parser.getspectrum(s)[0][0], minmz)
                maxmz = max(self.parser.getspectrum(s)[0][-1], maxmz)
            massrange = np.arange(minmz, maxmz, tol)
            gcoord2index = self._global2index()
            (minx, maxx), (miny, maxy), (minz, maxz), spectralength, mzidx = self.get_region_range(regID)
            regionshape = [maxx - minx + 1,
                           maxy - miny + 1]
            if maxz - minz + 1 > 1:
                regionshape.append(maxz - minz + 1)
            regionPixels = self.get_region_pixels(regID)
            array2D = np.zeros([len(regionPixels), len(massrange)], dtype=np.float32)
            # array3D = np.zeros(regionshape, dtype=np.float32)
            nS = np.random.randint(len(regionPixels))
            lCoorIdx = []
            for idx, coord in enumerate(tqdm(regionPixels, desc='binning', total=len(regionPixels))):
                xpos = coord[0] - minx
                ypos = coord[1] - miny
                lCoorIdx.append((xpos, ypos))
                mzs, ints = self.parser.getspectrum(gcoord2index[coord])
                for jdx, mz_value in enumerate(massrange):
                    min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
                    array2D[idx, jdx] = sum(ints[min_i:max_i + 1])  # = array3D[xpos, ypos, jdx]
                if idx == nS:
                    print("Plotting: #{}".format(nS))
                    rawVSprocessed(mzs, ints, massrange, array2D[nS, :], n_spec=nS, exprun='Resampling')
            if savedata:
                print("saving resampled data...")
                with h5py.File(resampfilename, 'w') as pfile:
                    pfile['spectra'] = array2D
                    pfile['mzrange'] = massrange
                    pfile['regionshape'] = regionshape
                    pfile['coordinates'] = lCoorIdx
        return array2D, massrange, regionshape, lCoorIdx

    def smooth_spectra(self, spectra, method="savgol", window_length=5, polyorder=2):
        assert (method in ["savgol", "gaussian"])
        nSpecs = spectra.shape[0]
        spectra_ = np.zeros_like(spectra)
        if method == "savgol":
            for s in range(nSpecs):
                smoothspec = signal.savgol_filter(spectra[s, :], window_length=window_length, polyorder=polyorder, mode='nearest')
                smoothspec[smoothspec < 0] = 0
                spectra_[s, :] = smoothspec
        elif method == "gaussian":
            for s in range(nSpecs):
                smoothspec = ndimage.gaussian_filter1d(spectra[s, :], sigma=window_length, mode='nearest')
                smoothspec[smoothspec < 0] = 0
                spectra_[s, :] = smoothspec
        return spectra_

    def ms_peak_picker_wrapper(self, rawspectra, step=0.01, snr=7, mode='centroid', int_thr_fact=1000, fit='quadratic',
                               precision=5, fwhm_expansion=1.2, savedir=None):
        """
        :param rawspectra: list/ndarray object with #spectrum x Nmz x Nint (N varies for processed spectra)
        :param step: upsampling step
        :param snr: SNR threshold
        :param mode: 'profile' or 'centroid'
        :param int_thr:
        :param fit:
        :param precision:
        :param fwhm_expansion: higher value increases peak width + height as intervals are summed

        :return: peakspectra -> 2d array #spectrum x nPeaks
                 peakmzs -> list m/zs where peaks were found

        e.g.:   ImzObj = Imzmlall(mspath)
                peakspectra, peakmzs = ImzObj.ms_peak_picker_wrapper(spectra)
        """
        binner = ms_peak_picker.scan_filter.LinearResampling(step)
        rebinned = [binner(*s) for s in rawspectra]
        intensity_grid = np.vstack([ri[1] for ri in rebinned])
        mz_grid, intensity_average = ms_peak_picker.average_signal(rawspectra, dx=step)
        intensity_average *= len(rawspectra)

        peak_list = ms_peak_picker.pick_peaks(mz_grid, intensity_average,
                                              signal_to_noise_threshold=snr, peak_mode=mode,
                                              intensity_threshold=int_thr_fact * len(rawspectra), fit_type=fit)
        peakmzs = [np.round(p.mz, precision) for p in peak_list]
        print("There will be #{} peaks...".format(len(peakmzs)))
        peak_ranges = [
            (
                p.mz - (p.full_width_at_half_max * fwhm_expansion),
                p.mz + (p.full_width_at_half_max * fwhm_expansion),
            )
            for p in peak_list]
        peak_indices = [(self._find_nearest(mz_grid, p[0]), self._find_nearest(mz_grid, p[1]))
                        for p in peak_ranges]
        peakspectra = []
        for spectrum in tqdm(intensity_grid, desc='aligning peaks...'):
            peaks = []
            for p in peak_indices:
                peaks.append(np.sum(spectrum[p[0]: p[1]]))
            peakspectra.append(peaks)
        peakspectra = np.array(peakspectra, dtype=np.float32)

        if os.path.isdir(savedir):
            peakfilename = os.path.join(savedir,'peakspectra_step{}_snr{}_int{}.h5'
                                        .format(str(step)[2:], snr, int_thr_fact))
            print("saving file in {}".format(peakfilename))
            with h5py.File(peakfilename, 'w') as pfile:  # saves the data
                pfile['peakspectra'] = peakspectra
                pfile['peakmzs'] = peakmzs
        return peakspectra, peakmzs

    def get_mean_abundance(self):
        """
        for all regions
        """
        labeled_array, num_features = self._get_regions()
        meanintensity = []
        for r in range(num_features):
            array2D, longestmz, regionshape, lCoorIdx = self.get_region_data(r + 1)
            meanintensity.append(np.mean(array2D, axis=0))
        return np.mean(np.array(meanintensity, dtype=np.float32), axis=0)

    def getionimage(self, regID, mz_value, tol=0.1, z=1, reduce_func=sum):
        """
        Get an image representation of the intensity distribution
        of the ion with specified m/z value.
        By default, the intensity values within the tolerance region are summed.
        :param mz_value:
            m/z value for which the ion image shall be returned
        :param tol:
            Absolute tolerance for the m/z value, such that all ions with values
            mz_value-|tol| <= x <= mz_value+|tol| are included. Defaults to 0.1
        :param z:
            z Value if spectrogram is 3-dimensional.
        :param reduce_func:
            the bahaviour for reducing the intensities between mz_value-|tol| and mz_value+|tol| to a single value. Must
            be a function that takes a sequence as input and outputs a number. By default, the values are summed.
        :return:
            numpy matrix with each element representing the ion intensity in this
            pixel. Can be easily plotted with matplotlib
        """
        tol = abs(tol)
        regionshape, localCoords = self.get_region_shape_coords(regID)
        globalCoords = self.get_region_pixels(regID)
        gcoord2index = self._global2index()
        im = np.zeros([regionshape[0], regionshape[1]])
        for i, ((r, c), (x, y, z_)) in enumerate(zip(localCoords, globalCoords)):
            if z_ == 0:
                UserWarning("z coordinate = 0 present, if you're getting blank images set getionimage(.., .., z=0)")
            if z_ == z:
                sIdx = gcoord2index[(x, y, z_)]
                mzs, ints = map(lambda p: np.asarray(p), self.parser.getspectrum(sIdx))
                min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
                im[r, c] = reduce_func(ints[min_i:max_i+1])     # y - 1, x - 1
        return im

    def get_ion_images(self, regID, peakspectra=None, peakmzs=None, **kwargs):
        """
        For visualization, to see if sufficient ion images could be generated...
        peak: plot peak images, default: True
        """
        if any(v is None for v in [peakspectra, peakmzs]):
            array2D, mzrange, regionshape, lCoorIdx = self.resample_region(regID, tol=0.02)
            peakspectra, peakmzs = self.peak_pick(array2D, refmz=mzrange)
        # array2D, longestmz, regionshape, lCoorIdx = self.get_region_data(regID)
        else:
            regionshape, lCoorIdx = self.get_region_shape_coords(regID)
        # peak3D = np.zeros([regionshape[0], regionshape[1], len(peakmzs)])
        # for idx, coord in enumerate(lCoorIdx):
        #     peak3D[coord[0], coord[1], :] = peakspectra[idx, :]
        peak3D = _2d_to_3d(peakspectra, regionshape, lCoorIdx)
        colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
        n_bin = 100
        if __name__ != '__main__':
            TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
            # print(TIME_STAMP)
            mtl.colormaps.register(LinearSegmentedColormap.from_list(name='{}_list'.format(TIME_STAMP), colors=colors, N=n_bin))
        imgN = 50   # how many images to take and plot
        if kwargs == 'top': # top variance images...
            peakvar = []
            for mz in range(len(peakmzs)):
                peakvar.append(np.std(peak3D[..., mz]))
            topmzInd = sorted(sorted(range(len(peakvar)), reverse=False, key=lambda sub: peakvar[sub])[-imgN:])
            # topmzInd = sorted(sorted(range(len(peakvar)), reverse=False, key=lambda sub: peakvar[sub])[-imgN-3000:-3000])
        elif kwargs == 'random':   # random
            topmzInd = np.round(np.linspace(0, len(peakmzs) - 1, imgN)).astype(int)
        else:
            topmzInd = np.arange(0, 50)
        Nr = 10
        Nc = 5
        heights = [regionshape[1] for r in range(Nr)]
        widths = [regionshape[0] for r in range(Nc)]
        fig_width = 5.  # inches
        fig_height = fig_width * sum(heights) / sum(widths)
        fig, axs = plt.subplots(Nr, Nc, figsize=(fig_width, fig_height), dpi=600, constrained_layout=True,
                                gridspec_kw={'height_ratios': heights})
        fig.suptitle('reg {}: ion images({})'.format(regID, kwargs), y=0.99)
        images = []
        pv = 0
        for r in range(Nr):
            for c in range(Nc):
                # Generate data with a range that varies from one plot to the next.
                images.append(axs[r, c].imshow(peak3D[..., topmzInd[pv]].T, origin='lower',
                                               cmap='{}_list'.format(TIME_STAMP)))  # 'RdBu_r')) #
                axs[r, c].label_outer()
                axs[r, c].set_axis_off()
                axs[r, c].set_title('{}'.format(peakmzs[topmzInd[pv]]), fontsize=5, pad=0.25)
                fig.subplots_adjust(top=0.95, bottom=0.02, left=0,
                                    right=1, hspace=0.14, wspace=0)
                pv += 1
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
        plt.show()

def umap_it(data, n_neighbors=12, n_components=3,
            min_dist=0.025, plot_umap=True, random_state=20210131):
    """
    from grid-parameter search in Hang Hu paper(visual plot)
    from grid-param Hang Hu paper(visual plot)
    :param data: standardized data to be umapped
    :param n_neighbors:
    :param n_components:
    :param min_dist:
    :param random_state:
    :return:
    """
    reducer = UMAP(n_neighbors=n_neighbors,
                   # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                   n_components=n_components,  # default 2, The dimension of the space to embed into.
                   metric='cosine',
                   # default 'euclidean', The metric to use to compute distances in high dimensional space.
                   n_epochs=1000,
                   # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
                   learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
                   init='spectral',
                   # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                   min_dist=min_dist,  # default 0.1, The effective minimum distance between embedded points.
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
                   random_state=random_state,
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
    data_umap = reducer.fit_transform(data)
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
    return data_umap

def hdbscan_it(data, HDBSCAN_soft = False, min_cluster_size = 250,
               min_samples=30, cluster_selection_method='eom'):
    """
    :param data:
    :param HDBSCAN_soft:
    :param min_cluster_size:
    :param min_samples:
    :param cluster_selection_method:
    :return:
    """
    if HDBSCAN_soft:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_method=cluster_selection_method, prediction_data=True) \
            .fit(data)
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        labels = np.argmax(soft_clusters, axis=1)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_method=cluster_selection_method).fit(data)
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
    plt.scatter(data[:, 0], data[:, 1], facecolors='None', edgecolors=cm.tab20(labels), alpha=0.9)
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
    chart(data, labels)
    return labels

def masterPlot(imageArray, valueList=None, oneImageshape=None, Title=None, Nc=3):
    """
    :param imageArray: could be list of row x col x n   ndarray
    :param valueList: list in str(list) format
    :param oneImageshape:
    :param Title:
    :param Nc:
    :return:
    """
    imageArray = imageArray.transpose(2, 0, 1)
    Nr = len(imageArray) // Nc + (len(imageArray) % Nc > 0)
    if oneImageshape == None:
        oneImageshape = np.shape(imageArray[0])
    heights = [oneImageshape[1] for r in range(Nr)]
    widths = [oneImageshape[0] for c in range(Nc)]
    fig_width = 5.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    fig, axes = plt.subplots(Nr, Nc, figsize=(fig_width, fig_height), dpi=600,
                             constrained_layout=True,
                             gridspec_kw={'height_ratios': heights})
    if Nr*Nc > len(imageArray):
        axesList = list(np.arange(Nr * Nc))
        imgNList = list(np.arange(len(imageArray)))
        new_list = list(set(axesList).difference(imgNList))
        for i in new_list:
            dr, dc = np.unravel_index(i, (Nr, Nc), 'C')
            # print("dr, dc", dr, dc)
            fig.delaxes(axes[dr, dc])
    if Title == None:
        Title = 'Plot'
    if valueList is None:
        valueList = np.arange(1, len(imageArray) + 1)
    fig.suptitle('{}'.format(Title), y=0.99)
    images = []
    for i, (image, ax) in enumerate(tqdm(zip(imageArray, axes.ravel()), total=len(imageArray))):
        images.append(ax.imshow(image.T, origin='lower', cmap='msml_list'))
        ax.label_outer()
        ax.set_axis_off()
        # if valueList == None:
        ax.set_title("{}".format(valueList[i]), fontsize=5, pad=0.25)
        # else:
        # ax.set_title((valueList[i]), fontsize=5, pad=0.25)
        ax.set_xlabel("")
        # fig.get_legend().remove()
        fig.subplots_adjust(top=0.90, bottom=0.02, left=0,
                            right=1, hspace=0.14, wspace=0)


    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())
                im.set_tight_layout('tight')

    for im in images:
        im.callbacks.connect('changed', update)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.55])  # left, bottom, width, height
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.5)
    cbar.ax.tick_params(labelsize=5) #the tick size on colorbar
    plt.show()

def kdemzs(peakmzs, bw): #TODO: implement kneedle/find knee
    if bw is None:
        cl = []
        for bw in range(1, 100):
            nClusters, _ = kdemzs(peakmzs, bw)
            cl.append(nClusters)

        fig, ax = plt.subplots(dpi=100)
        ax.plot(np.arange(1, 100), cl)
        ax.set_xlim(0, 104)
        ax.set_xticks(np.arange(1, 101, 6))
        ax.set_xlabel("bandwidth", fontsize=10)
        ax.set_ylabel("#clusters", fontsize=10)
        ax.set_title("Kernel Density Estimation - peak m/z s", fontsize=15)
        plt.show()
    else:
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(peakmzs)  # 30
        s = np.linspace(min(peakmzs), max(peakmzs))
        e = kde.score_samples(s.reshape(-1, 1))

        mi, ma = signal.argrelextrema(e, np.less)[0], signal.argrelextrema(e, np.greater)[0]
        mi_ = np.sort(np.append(mi, [0, len(s) - 1]))

        center_ = s[ma].reshape(-1)

        cluster_centers = np.asarray([find_nearest(peakmzs, i) for i in center_]).reshape(-1)
        print("there are {} clusters(m/z s) with centers at: {}".format(len(cluster_centers), cluster_centers))
        clusters = []
        for mdx in range(len(mi_) - 1):
            cluster = peakmzs[np.where(
                (peakmzs >= s[mi_][mdx]) & (peakmzs < s[mi_][mdx + 1]))]
            # print(len(cluster), "\n", cluster)
            clusters.append(cluster)
        plt.plot(s, e, 'orange')
        plt.vlines(cluster_centers, ymin=min(e), ymax=max(e), colors='blue')
        plt.show()
    return len(cluster_centers), cluster_centers

def displayImage(matrix):
    plt.imshow(matrix.T, origin='lower', cmap='msml_list')
    plt.colorbar()
    plt.show()

def boxplotit(data, labels, Title):
    """
    example: boxplotit([sparse_coherence, dense_coherence], ['sparse', 'dense'], 'spatial coherence')
    :param data: list
    :param labels: list
    :param Title: str
    :return: boxplots
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)
    # labels = ['old', 'new']
    colorlist = ['maroon', 'darkblue', 'orangered', 'olive', 'sienna']
    colornames = colorlist[:len(labels)]
    medianprops = dict(linestyle='-', linewidth=2.5, color='yellow') #next(iter(colornames))) #
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='brown') #next(iter(colornames))) #
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
    ax1.set_title('{}'.format(Title), fontsize=16)
    medians = [bplot['medians'][i].get_ydata()[0] for i in range(len(labels))]
    pos = np.arange(len(labels)) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    for tick, label in zip(range(len(labels)), ax1.get_xticklabels()):
        ax1.text(pos[tick], .97, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', fontsize=10,
                 weight='bold',
                 color=colornames[tick]) #'firebrick')  # )
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    for patch, color in zip(bplot['boxes'], colornames):
        patch.set_facecolor(color) #'slategrey')
        patch.set_alpha(1)
    fig.show()

def saveimages(images, mzs, directory):
    """
    save images in drive
    :param images: row x column x n
    :param mzs: list, 1D array
    :param directory: where to save in the drive
    :return:
    """
    assert (images.shape[-1] == len(mzs))

    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    if not os.path.isdir(directory):
        os.mkdir(directory)
    for mz in range(images.shape[-1]):
        filename = os.path.join(directory, '{}.png'.format(mzs[mz]))
        norm_img = np.uint8(cv2.normalize(images[..., mz], None, 0, 255, cv2.NORM_MINMAX))
        image = cv2.rotate(norm_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.applyColorMap(image, get_mpl_colormap('msml_list'))
        # image = cv.LUT(image, lut)
        cv2.imwrite(filename, image)
    return

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
            raise ValueError(f'Input shape of {peaks.shape} does not match expected shape '
                'for spectrum [n_peaks, 2].')
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

class BinningVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer based on binning MALDI-TOF spectra.
    Attributes:
        bin_edges_: Edges of the bins derived after fitting the transformer.
    """
    _required_parameters = ['n_bins']

    def __init__(
        self,
        n_bins,
        min_bin=float('inf'),
        max_bin=float('-inf'),
        n_jobs=None
    ):
        """Initialize BinningVectorizer.

        Args:
            n_bins: Number of bins to bin the inputs spectra into.
            min_bin: Smallest possible bin edge.
            max_bin: Largest possible bin edge.
            n_jobs: If set, uses parallel processing with `n_jobs` jobs
        """
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bin_edges_ = None
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit transformer, derives bins used to bin spectra."""
        combined_times = np.concatenate(    # just serialize all mzs in all spec
            [spectrum[:, 0] for spectrum in X], axis=0)
        min_range = min(self.min_bin, np.min(combined_times))
        max_range = max(self.max_bin, np.max(combined_times))
        _, self.bin_edges_ = np.histogram(
            combined_times, self.n_bins, range=(min_range, max_range))
        return self

    def transform(self, X):
        """Transform list of spectra into vector using bins.
        Args:
            X: List of MALDI-TOF spectra
        Returns:
            2D numpy array with shape [n_instances x n_bins]
        """
        if self.n_jobs is None:
            output = [self._transform(spectrum) for spectrum in X]
        else:
            output = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._transform)(s) for s in X
            )

        return np.stack(output, axis=0)

    def _transform(self, spectrum):
        times = spectrum[:, 0]  ## mz, time of flight
        indices = np.digitize(times, self.bin_edges_, right=True)
        # Drops all instances which are outside the defined bin
        # range.
        valid = (indices >= 1) & (indices <= self.n_bins)   # boolean
        # print("valid", valid)
        spectrum = spectrum[valid]
        # Need to update indices to ensure that the first bin is at
        # position zero.
        indices = indices[valid] - 1
        identity = np.eye(self.n_bins)

        vec = np.sum(identity[indices] * spectrum[:, 1][:, np.newaxis], axis=0)
        return vec

    def detail_(self, x):
        print("\n")
        print(x)
        print(type(x))
        import matplotlib.pyplot as plt
        try:
            plt.plot(x)
            plt.show()
        except:
            pass

class Binning2(object):
    """
    given the imze object should create 3D matrix(spatial based) or 2D(spectrum based)
    spectrum: array with 2 vectors, one of abundance(1), other with m/z values(0)
    n_bins: number of bins/samples to be digitized
    plotspec: to plot the new binned spectrum, default--> True
    """
    def __init__(self, imzObj, regionID, n_bins, plotspec=False):   #binned=True
        self.imzObj = imzObj
        self.regionID = regionID
        if n_bins is None:
            self.n_bins = len(imzObj.mzValues) + 1
        else:
            self.n_bins = n_bins
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
            # coordList.append(coord)         # global coordinates
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

    def getUnbinned(self):
        regInd = self.imzObj.get_region_indices(self.regionID)
        spec_data = np.zeros([len(regInd), len(self.imzObj.mzValues)]) ## (3435, 1332)
        spec_array = self.imzObj.get_region_array(self.regionID)
        xr, yr, zr, _ = self.imzObj.get_region_range(self.regionID)
        coordList = []
        for idx, coord in enumerate(regInd):
            spectrum = self.imzObj.parser.getspectrum(self.imzObj.coord2index.get(coord))  # [0]
            spec_data[idx] = spectrum[1]
            xpos = coord[0] - xr[0]
            ypos = coord[1] - yr[0]
            spec_array[xpos, ypos] = spectrum[1]
            coordList.append((xpos, ypos))
        return spec_array, spec_data, coordList

    def MaldiTofBinning(self):
        regInd = self.imzObj.get_region_indices(self.regionID)
        regSpecData = []
        regSpecCoor = []
        for idx, coord in tqdm(enumerate(regInd), desc='#binning%'):
            spectrum = self.imzObj.parser.getspectrum(self.imzObj.coord2index.get(coord))  # [0]
            maldispec = []
            [maldispec.append([mz, ab]) for mz, ab in zip(spectrum[0], spectrum[1])]  # , spectrum[1]:
            regSpecData.append(MaldiTofSpectrum(maldispec))
            regSpecCoor.append((coord[0], coord[1]))
        vectorizer = BinningVectorizer(self.n_bins, n_jobs=-1) #min_bin=510, max_bin=1800 #TODO: fix the max-min auto
        vectorized = vectorizer.fit_transform(regSpecData)

        # spec_array #self.imzObj.get_region_array(self.regionID)
        xr, yr, zr, _ = self.imzObj.get_region_range(self.regionID)
        # self.
        spec_array = np.zeros([self.imzObj.get_region_shape(self.regionID)[0],
                               self.imzObj.get_region_shape(self.regionID)[1], self.n_bins])
        print(spec_array.shape)
        coordList = []
        for idx, coord in enumerate(regSpecCoor):
            xpos = coord[0] - xr[0]
            ypos = coord[1] - yr[0]
            spec_array[xpos, ypos] = vectorized[idx]
            coordList.append((xpos, ypos))
        return spec_array, np.array(vectorized), coordList

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
    return array[idx] #[0]

def normalize_spectrum(spectrum, normalize=None, max_region_value=None):
    """Normalizes a single spectrum.
    Args:
        spectrum (numpy.array): Spectrum to normalize.
        normalize (str, optional): Normalization method. Must be "max_intensity_spectrum", "max_intensity_region", "vector". Defaults to None.\n
            - "max_intensity_spectrum": divides the spectrum by the maximum intensity value.\n
            - "max_intensity_region"/"max_intensity_all_regions": divides the spectrum by custom max_region_value.\n
            - "vector": divides the spectrum by its norm.\n
            - "tic": divides the spectrum by its TIC (sum).\n
        max_region_value (int/float, optional): Value to normalize to for max-region-intensity norm. Defaults to None.

    Returns:
        numpy.array: Normalized spectrum.
    """
    assert (normalize in [None, "zscore", "tic", "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector"])
    retSpectrum = np.array(spectrum, copy=True)

    # if normalize in ["max_intensity_region", "max_intensity_all_regions"]:
    #     assert(max_region_value != None)

    if normalize == "max_intensity_spectrum":
        retSpectrum = retSpectrum / np.max(retSpectrum)
        return retSpectrum

    elif normalize == "max_intensity_region": #, "max_intensity_all_regions"]:
        retSpectrum = retSpectrum / np.max(spectrum)
        return retSpectrum

    elif normalize == "tic":
        specSum = sum(retSpectrum)
        if specSum > 0:
            retSpectrum /= specSum #(retSpectrum / specSum)*len(retSpectrum)
        return retSpectrum #* len(retSpectrum)

    elif normalize in ["zscore"]:
        lspec = list(retSpectrum)
        nlspec = list(-retSpectrum)
        retSpectrum = np.array(stats.zscore(lspec + nlspec, nan_policy="omit")[:len(lspec)])
        retSpectrum = np.nan_to_num(retSpectrum)
        assert(len(retSpectrum) == len(lspec))
        return retSpectrum

    elif normalize == ["vector"]:
        slen = np.linalg.norm(retSpectrum)
        if slen < 0.01:
            retSpectrum = retSpectrum * 0
        else:
            retSpectrum = retSpectrum / slen
        #with very small spectra it can happen that due to norm the baseline is shifted up!
        retSpectrum[retSpectrum < 0.0] = 0.0
        retSpectrum = retSpectrum - np.min(retSpectrum)
        if not np.linalg.norm(retSpectrum) <= 1.01:
            print(slen, np.linalg.norm(retSpectrum))
        return retSpectrum

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
    mspath: path to .imzML file or .mat file where the array, spectra and coordinates are saved.
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
    save_rseg = False
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
    if os.path.isfile(regname):
        matr = loadmat(regname)
        regArr = matr['array']
        regSpec = matr['spectra']
        regCoor = matr['coordinates']
    else:
        ImzObj = IMZMLExtract(mspath)
        # BinObj = Binning2(ImzObj, regID)
        regArr, regSpec, regCoor = Binning2(ImzObj, regID, n_bins=1400).MaldiTofBinning()   #BinObj.getBinMat()
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
        regname = os.path.join(regDir, '{}_reg_{}_{}.mat'.format(filename, regID, downsamp_i))
        matr = {"spectra": regSpec, "array": regArr, "coordinates": regCoor, "info": "after peakpicking in Cardinal"}
        savemat(regname, matr)
    # +------------------------------+
    # |   normalize, standardize     |
    # +------------------------------+
    reg_norm = np.zeros_like(regSpec)
    # _, reg_smooth_, _ = bestWvltForRegion(regSpec, bestWvlt=None, smoothed_array=True, plot_fig=False)
    for s in range(0, nSpecs):
        reg_norm[s, :] = normalize_spectrum(regSpec[s, :], normalize='tic')     #reg_smooth_
        # reg_norm_ = _smooth_spectrum(regSpec[s, :], method='savgol', window_length=5, polyorder=2)
        reg_norm[s, :] = _smooth_spectrum(reg_norm[s, :], method='savgol', window_length=5, polyorder=2)
        # printStat(data_norm)
    reg_norm_ss = makeSS(reg_norm).astype(np.float64)
    # reg_norm_ss = StandardScaler().fit_transform(reg_norm)
    # +----------------+
    # |  plot spectra  |
    # +----------------+
    if plot_spec:
        nS = np.random.randint(nSpecs)
        fig, ax = plt.subplots(5, 1, figsize=(16, 10), dpi=200)
        ax[0].plot(regSpec[nS, :])
        ax[0].set_title("raw spectrum")
        ax[1].plot(reg_norm[nS, :])
        ax[1].set_title("'tic' norm")
        ax[2].plot(reg_norm_ss[nS, :])
        ax[2].set_title("standardized")
        ax[3].plot(np.mean(regSpec, axis=0))
        ax[3].set_title("mean spectra(region {})".format(regID))
        # ax[4].plot(reg_smooth_[nS, :])
        # ax[4].set_title("Smoothed...")
        plt.suptitle("Processing comparison of Spec #{}".format(nS))
        plt.show()

    data = copy.deepcopy(reg_norm_ss)
    # +------------+
    # |    PCA     |
    # +------------+
    pca = PCA(random_state=RandomState)     # pca object
    pcs = pca.fit_transform(data)   # (4587, 2000)
    pca_range = np.arange(1, pca.n_components_, 1)
    print(">> PCA: number of components #{}".format(pca.n_components_))
    # printStat(pcs)
    evr = pca.explained_variance_ratio_
    # print(evr)
    evr_cumsum = np.cumsum(evr)
    # print(evr_cumsum)
    cut_evr = find_nearest(evr_cumsum, threshold)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
    df_pca = pd.DataFrame(data=pcs[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])

    # # +------------------------------+
    # # |   Agglomerating Clustering   |
    # # +------------------------------+
    nCl = 7     # todo: how?
    agg = AgglomerativeClustering(n_clusters=nCl)
    assignment = agg.fit_predict(pcs)  # on pca
    mglearn.discrete_scatter(regCoor[:, 0], regCoor[:, 1], assignment, labels=np.unique(assignment))
    plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
    plt.title("Agglomerative Clustering")
    plt.show()
    #
    # if plot_pca:
    #     MaxPCs = nPCs + 5
    #     fig, ax = plt.subplots(figsize=(20, 8), dpi=200)
    #     ax.bar(pca_range[0:MaxPCs], evr[0:MaxPCs] * 100, color="steelblue")
    #     ax.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
    #     ax.set_xlabel('Principal component number', fontsize=30)
    #     ax.set_ylabel('Percentage of \nvariance explained', fontsize=30)
    #     ax.set_ylim([-0.5, 100])
    #     ax.set_xlim([-0.5, MaxPCs])
    #     ax.grid("on")
    #
    #     ax2 = ax.twinx()
    #     ax2.plot(pca_range[0:MaxPCs], evr_cumsum[0:MaxPCs] * 100, color="tomato", marker="D", ms=7)
    #     ax2.scatter(nPCs, cut_evr * 100, marker='*', s=500, facecolor='blue')
    #     ax2.yaxis.set_major_formatter(mtl.ticker.PercentFormatter())
    #     ax2.set_ylabel('Cumulative percentage', fontsize=30)
    #     ax2.set_ylim([-0.5, 100])
    #
    #     # axis and tick theme
    #     ax.tick_params(axis="y", colors="steelblue")
    #     ax2.tick_params(axis="y", colors="tomato")
    #     ax.tick_params(size=10, color='black', labelsize=25)
    #     ax2.tick_params(size=10, color='black', labelsize=25)
    #     ax.tick_params(width=3)
    #     ax2.tick_params(width=3)
    #
    #     ax = plt.gca()  # Get the current Axes instance
    #
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax.spines[axis].set_linewidth(3)
    #
    #     plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
    #     plt.show()
    #
    #     plt.figure(figsize=(12, 10), dpi=200)
    #     # plt.scatter(df_pca.PC_1, df_pca.PC_2, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
    #     # plt.scatter(df_pca.PC_1, df_pca.PC_2, c=assignment, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
    #     mglearn.discrete_scatter(df_pca.PC_1, df_pca.PC_2, assignment, alpha=0.5) #, labels=np.unique(assignment))
    #     plt.xlabel('PC1 ({}%)'.format(round(evr[0] * 100, 2)), fontsize=30)
    #     plt.ylabel('PC2 ({}%)'.format(round(evr[1] * 100, 2)), fontsize=30)
    #     plt.tick_params(size=10, color='black')
    #     # tick and axis theme
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
    #     ax = plt.gca()  # Get the current Axes instance
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax.spines[axis].set_linewidth(2)
    #     ax.tick_params(width=2)
    #     plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
    #     plt.show()

    # +-------------------+
    # |   HC_clustering   |
    # +-------------------+
    HC_method = 'ward'
    HC_metric = 'euclidean'
    Y = sch.linkage(df_pca.values, method=HC_method, metric=HC_metric)
    Z = sch.dendrogram(Y, no_plot=True)
    HC_idx = Z['leaves']
    HC_idx = np.array(HC_idx)

    # plot it
    thre_dist = 375  #TODO: how to fix it?
    plt.figure(figsize=(15, 10))
    Z = sch.dendrogram(Y, color_threshold=thre_dist)
    plt.title('hierarchical clustering of ion images \n method: {}, metric: {}, threshold: {}'.format(
        HC_method, HC_metric, thre_dist))

    ## 2. sort features with clustering results
    features_modi_sorted = df_pca.values[HC_idx]

    # plot it
    fig = plt.figure(figsize=(10, 10))
    axmatrix = fig.add_axes([0.10, 0, 0.80, 0.80])
    im = axmatrix.matshow(features_modi_sorted, aspect='auto', origin='lower', cmap=cm.YlGnBu, interpolation='none')
    fig.gca().invert_yaxis()

    # colorbar
    axcolor = fig.add_axes([0.96, 0, 0.02, 0.80])
    cbar = plt.colorbar(im, cax=axcolor)
    axcolor.tick_params(labelsize=10)
    plt.show()

    #TODO: fix here...

    # from scipy.cluster.hierarchy import fcluster
    # HC_labels = fcluster(Y, thre_dist, criterion='distance')
    #
    # # print("HC_labels >> ", len(np.unique(HC_labels)), HC_labels)
    # # prepare label data
    # elements, counts = np.unique(HC_labels, return_counts=True)
    # # print("elements >> ", elements, counts)     # [1 2 3 4 5]
    # ### PCA loadings
    # loadings1 = pca.components_.T    # loadings.shape > (2000, 3)
    # # loadings2 = pca.components_
    # # print("loadings >>", loadings, loadings.shape)
    # # sum of squared loadings
    # print("pca.components_ >> ", np.min(pca.components_),np.max(pca.components_), np.shape(pca.components_))
    # SSL1 = np.sum(loadings1 ** 2, axis=1)    # SSL >> (2000,) 2000
    # # SSL2 = np.sum(loadings2 ** 2, axis=1)  # SSL >> (2000,) 2000
    # # print("SSL >>", SSL.shape, len(SSL))
    # for label in elements:
    #     idx = np.where(HC_labels == label)[0]
    #     print("idx >>", idx)
    #     # total SSL
    #     total_SSL = np.sum(SSL1[idx])
    #     # imgs in the cluster
    #     # current_cluster = imgs_std[idx]
    #     # average img
    #     # mean_img = np.mean(imgs_std[idx], axis=0)
    #
    #     # accumulate data
    #     # total_SSLs.append(total_SSL)
    #     # mean_imgs.append(mean_img)


    # +------------------+
    # |      UMAP        |
    # +------------------+
    u_neigh = 12    # from grid-parameter search in Hang Hu paper(visual plot)
    u_comp = 3
    u_min_dist = 0.025  # from grid-param Hang Hu paper(visual plot)
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
    data_umap = reducer.fit_transform(data) #df_pca.values)  # on pca
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
    # sarray = nnPixelCorrect(sarray, 20, 3)  # noisy pixel is labeled as 19 by hdbscan
    try:
        sarray = nnPixelCorrect(sarray, 20, 3)  # noisy pixel is labeled as 19 by hdbscan
    except:
        pass
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
    if save_rseg:
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
        ax.set_title('reg{}: {}'.format(regID, columnName), fontsize=15, loc='center')
        plt.show()
        if save_rseg:
            namepy = os.path.join(regDir, '{}_{}.npy'.format(exprun, columnName))
            np.save(namepy, sarray1)
    return

def msmlfunc4(mspath, regID, threshold, exprun):
    """
    performs ML on resampled and peak picked spectra
    """
    plot_spec = True
    plot_pca = True
    plot_umap = True
    save_rseg = True
    RandomState = 20210131
    # +------------------------------------+
    # |     read data and save region      |
    # +------------------------------------+
    dirname = os.path.dirname(mspath)
    basename = os.path.basename(mspath)
    filename, ext = os.path.splitext(basename)
    regDir = os.path.join(dirname, 'reg_{}'.format(regID))
    if not os.path.isdir(regDir):
        os.mkdir(regDir)
    regname = os.path.join(regDir, '{}_reg_{}_{}.h5'.format(filename, regID, exprun))
    if os.path.isfile(regname):
        f = h5py.File(regname, 'r')
        spectra = f['spectra']
        localCoor = f['coordinates']
        regionshape = f['regionshape']
        peakmzs = f['peakmzs']
    else:
        ImzObj = ImzmlAll(mspath)
        spectra, refmz, regionshape, localCoor = ImzObj.get_region_data(regID, whole=False)
        spectra_smoothed = ImzObj.smooth_spectra(spectra, window_length=9, polyorder=2)
        spectra, peakmzs = ImzObj.peak_pick(spectra_smoothed, refmz)
        with h5py.File(regname, 'w') as pfile:
            pfile['spectra'] = spectra
            pfile['coordinates'] = localCoor
            pfile['regionshape'] = regionshape
            pfile['peakmzs'] = peakmzs
    nSpecs, nBins = spectra.shape
    # +------------------------------+
    # |   normalize, standardize     |
    # +------------------------------+
    # _, reg_smooth_, _ = bestWvltForRegion(regSpec, bestWvlt='db8', smoothed_array=True, plot_fig=False)
    reg_norm = np.zeros_like(spectra)
    for s in range(nSpecs):
        reg_norm[s, :] = normalize_spectrum(spectra[s, :], normalize='max_intensity_region')     #reg_smooth_
    reg_norm_ss = makeSS(reg_norm).astype(np.float64)
    # reg_norm_ss = SS(with_mean=True, with_std=True).fit_transform(reg_norm)
    # +----------------+
    # |  plot spectra  |
    # +----------------+
    if plot_spec:
        nS = np.random.randint(nSpecs)
        fig, ax = plt.subplots(4, 1, figsize=(16, 10), dpi=200)
        ax[0].plot(spectra[nS, :])
        ax[0].set_title("raw spectrum")
        ax[1].plot(reg_norm[nS, :])
        ax[1].set_title("max norm")
        ax[2].plot(reg_norm_ss[nS, :])
        ax[2].set_title("standardized")
        ax[3].plot(np.mean(spectra, axis=0))
        ax[3].set_title("mean spectra(region {})")
        # ax[4].plot(reg_smooth_[nS, :])
        # ax[4].set_title("Smoothed...")
        plt.suptitle("Processing comparison of Spec #{}".format(nS))
        plt.show()

    data = copy.deepcopy(reg_norm_ss)
    # +------------+
    # |    PCA     |
    # +------------+
    pca = PCA(random_state=RandomState)  # pca object
    pcs = pca.fit_transform(data)  # (4587, 2000)
    # pcs=pca.fit_transform(oldLipid_mm_norm)
    pca_range = np.arange(1, pca.n_components_, 1)
    print(">> PCA: number of components #{}".format(pca.n_components_))
    # printStat(pcs)
    evr = pca.explained_variance_ratio_
    # print(evr)
    evr_cumsum = np.cumsum(evr)
    # print(evr_cumsum)
    cut_evr = find_nearest(evr_cumsum, threshold)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
    df_pca = pd.DataFrame(data=pcs[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])

    # +------------------------------+
    # |   Agglomerating Clustering   |
    # +------------------------------+
    nCl = 4
    agg = AgglomerativeClustering(n_clusters=nCl)
    assignment = agg.fit_predict(pcs)  # on pca
    mglearn.discrete_scatter(np.array([i[0] for i in localCoor]),
                             np.array([i[1] for i in localCoor]), assignment, labels=np.unique(assignment))
    plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
    plt.title("Agglomerative Clustering")
    plt.show()
    seg = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
    for idx, coor in enumerate(localCoor):
        seg[coor[0], coor[1]] = assignment[idx] + 1
    plt.imshow(seg)
    plt.title("Agglomerative segmentation")
    plt.show()

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

        plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
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
        plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
        ax = plt.gca()  # Get the current Axes instance
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2)
        plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
        plt.show()

    # +-------------------+
    # |   HC_clustering   |
    # +-------------------+
    HC_method = 'ward'
    HC_metric = 'euclidean'
    fig = plt.figure(figsize=(15, 15), dpi=300)
    # plot dendogram
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    Y = sch.linkage(df_pca.values, method=HC_method, metric=HC_metric)
    thre_dist = 375  # TODO: how to fix it?
    Z = sch.dendrogram(Y, color_threshold=thre_dist, orientation='left')
    fig.gca().invert_yaxis()
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    # plot matrix/feature(sorted)
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8]) #left, bottom, width, height
    HC_idx = np.array(Z['leaves'])
    features_modi_sorted = df_pca.values[HC_idx]
    im = axmatrix.matshow(features_modi_sorted, aspect='auto', origin='lower', cmap=cm.YlGnBu, interpolation='none')
    fig.gca().invert_yaxis()
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # colorbar
    axcolor = fig.add_axes([0.92, 0.1, 0.02, 0.80])
    axcolor.tick_params(labelsize=10)
    plt.colorbar(im, cax=axcolor)
    fig.suptitle('hierarchical clustering of ion images \n method: {}, metric: {}, threshold: {}'.format(
        HC_method, HC_metric, thre_dist), fontsize=16)
    fig.show()

    HC_labels = sch.fcluster(Y, thre_dist, criterion='distance')
    # # prepare label data
    elements, counts = np.unique(HC_labels, return_counts=True)
    print(elements, counts)
    sarray1 = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
    for idx, coor in enumerate(localCoor):
        sarray1[coor[0], coor[1]] = HC_labels[idx]

    fig, ax = plt.subplots(figsize=(6, 8))
    sarrayIm = ax.imshow(sarray1)
    fig.colorbar(sarrayIm)
    ax.set_title('reg{}: HC Clustering'.format(regID), fontsize=15, loc='center')
    plt.show()

    if __name__ != '__main__':  #TODO: Fix PCA loadings and ion imaging
        # loadings = pca.components_.T
        loadings = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(x+1) for x in range(nBins)])
        print("loadings: ", loadings) #['PC897'])
        SL = loadings**2
        print("SL \n", SL)
        SSL = np.sum(SL, axis=1)
        print(SSL)

        print(loadings.shape)
        SSL = np.sum(loadings ** 2, axis=1)
        print(SSL.shape)
        mean_imgs = []
        total_SSLs = []
            # # img_std =
            # for label in elements:
            #     idx = np.where(HC_labels == label)[0]
            #
            #     # total SSL
            #     total_SSL = np.sum(SSL[idx])
            #     print(total_SSL)
            #     # imgs in the cluster
            #     current_cluster = imgs_std[idx]
            #     # average img
            #     mean_img = np.mean(imgs_std[idx], axis=0)
            #
            #     # accumulate data
            #     total_SSLs.append(total_SSL)
            #     mean_imgs.append(mean_img)

    # +------------------+
    # |      UMAP        |
    # +------------------+
    u_neigh = 12    # from grid-parameter search in Hang Hu paper(visual plot)
    u_comp = 3
    u_min_dist = 0.025  # from grid-param Hang Hu paper(visual plot)
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
    data_umap = reducer.fit_transform(data) #df_pca.values)  # on pca
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

    # +--------------------------------------+
    # |   Segmentation on PCA+UMAP+HDBSCAN   |
    # +--------------------------------------+
    sarray = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
    for idx, coor in enumerate(localCoor):
        sarray[coor[0], coor[1]] = labels[idx] + 1
    try:
        sarray = nnPixelCorrect(sarray, 20, 3)  # noisy pixel is labeled as 19 by hdbscan
    except:
        pass
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
    print("Consider for gmm >>", df_pca_umap.columns[0:nPCs+reducer.n_components])
    for i in range(repeat):  # may repeat several times
        for j in range(n_component.shape[0]):  # ensemble with different n_component value
            StaTime = time.time()
            gmm = GMM(n_components=n_component[j], max_iter=5000)  # max_iter does matter, no random seed assigned
            labels = gmm.fit_predict(df_pca_umap.iloc[:, 0:nPCs+reducer.n_components])     #todo data_umap
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
    if save_rseg:
        savecsv = os.path.join(regDir, '{}.csv'.format(exprun))
        df_pca_umap_hdbscan_gmm.to_csv(savecsv, index=False, sep=',')

    nGs = retrace_columns(df_pca_umap_hdbscan_gmm.columns.values, 'gmm')
    df_gmm_labels = df_pca_umap_hdbscan_gmm.iloc[:, -nGs:]
    # print("gmm label: ", nGs)

    for (columnName, columnData) in df_gmm_labels.iteritems():
        print('Column Name : ', columnName)
        print('Column Contents : ', np.unique(columnData.values))
        # regInd = ImzObj.get_region_indices(regID)
        # xr, yr, zr, _ = ImzObj.get_region_range(regID)
        # xx, yy, _ = ImzObj.get_region_shape(regID)
        sarray1 = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
        for idx, coor in enumerate(localCoor):
            sarray1[coor[0], coor[1]] = columnData.values[idx] + 1
        fig, ax = plt.subplots(figsize=(6, 8))
        sarrayIm = ax.imshow(sarray1)
        fig.colorbar(sarrayIm)
        ax.set_title('reg{}: {}'.format(regID, columnName), fontsize=15, loc='center')
        plt.show()
        if save_rseg:
            namepy = os.path.join(regDir, '{}_{}.npy'.format(exprun, columnName))
            np.save(namepy, sarray1)
    return

def msmlfunc5(mspath, regID, threshold, exprun, save_rseg=False):
    """
    performs ML on resampled and peak picked spectra based on tolerance
    without interpolation...
    """
    colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
    n_bin = 100
    mtl.colormaps.register(LinearSegmentedColormap.from_list(name='simple_list', colors=colors, N=n_bin))
    plot_spec = True
    plot_pca = True
    plot_umap = True
    # save_rseg = False
    RandomState = 20210131
    # +------------------------------------+
    # |     read data and save region      |
    # +------------------------------------+
    dirname = os.path.dirname(mspath)
    basename = os.path.basename(mspath)
    filename, ext = os.path.splitext(basename)
    regDir = os.path.join(dirname, 'reg_{}'.format(regID))
    if not os.path.isdir(regDir):
        os.mkdir(regDir)
    regname = os.path.join(regDir, '{}_reg_{}_{}.h5'.format(filename, regID, exprun))
    if os.path.isfile(regname):
        f = h5py.File(regname, 'r')
        spectra = f['spectra']
        localCoor = f['coordinates']
        regionshape = f['regionshape']
        peakmzs = f['peakmzs']
    else:
        ImzObj = ImzmlAll(mspath)
        spectra, refmz, regionshape, localCoor = ImzObj.resample_region(regID, tol=0.02, savedata=True)
        spectra_smoothed = ImzObj.smooth_spectra(spectra, window_length=9, polyorder=2)
        spectra, peakmzs = ImzObj.peak_pick(spectra_smoothed, refmz)
        with h5py.File(regname, 'w') as pfile:
            pfile['spectra'] = spectra
            pfile['coordinates'] = localCoor
            pfile['regionshape'] = regionshape
            pfile['peakmzs'] = peakmzs
    nSpecs, nBins = spectra.shape
    # +------------------------------+
    # |   normalize, standardize     |
    # +------------------------------+
    spectra = np.array(spectra)     # nPixel x nMz
    spec_norm = (spectra - spectra.min()) / (spectra.max() - spectra.min())
    def _2d_to_3d(array2d, Coord, regionshape):
        nPixels, nMz = array2d.shape
        array3d = np.zeros([nMz, regionshape[0], regionshape[1]])
        for idx, c in enumerate(Coord):
            array3d[:, c[0], c[1]] = array2d[idx, :]
        return array3d
    images = _2d_to_3d(spec_norm, localCoor, regionshape)   #3D ->  nMz x nPixelx x nPixely
    mz_features = images.reshape((len(images), -1))
    mz_features = SS().fit_transform(mz_features)
    # _, reg_smooth_, _ = bestWvltForRegion(regSpec, bestWvlt='db8', smoothed_array=True, plot_fig=False)
    # reg_norm = np.zeros_like(spectra)
    # for s in range(nSpecs):
        # reg_norm[s, :] = normalize_spectrum(spectra[s, :], normalize='tic')  # max_intensity_spectrum   #reg_smooth_
    # reg_norm_ss = makeSS(reg_norm).astype(np.float64)
    # reg_norm_ss = SS(with_mean=True, with_std=True).fit_transform(reg_norm)
    # +----------------+
    # |  plot spectra  |
    # +----------------+
    # if plot_spec:
    #     nS = np.random.randint(nSpecs)
    #     fig, ax = plt.subplots(4, 1, figsize=(16, 10), dpi=200)
    #     ax[0].plot(spectra[nS, :])
    #     ax[0].set_title("raw spectrum")
    #     ax[1].plot(reg_norm[nS, :])
    #     ax[1].set_title("max norm")
    #     ax[2].plot(reg_norm_ss[nS, :])
    #     ax[2].set_title("standardized")
    #     ax[3].plot(np.mean(spectra, axis=0))
    #     ax[3].set_title("mean spectra(region {})")
    #     # ax[4].plot(reg_smooth_[nS, :])
    #     # ax[4].set_title("Smoothed...")
    #     plt.suptitle("Processing comparison of Spec #{}".format(nS))
    #     plt.show()
    pixel_features = copy.deepcopy(mz_features.T)    # nPixel x nMz
    # +------------+
    # |    PCA     |
    # +------------+
    pca = PCA(random_state=RandomState)  # pca object n_components=100,
    pcs = pca.fit_transform(pixel_features)  # (4587, 2000)
    # pcs=pca.fit_transform(oldLipid_mm_norm)
    pca_range = np.arange(1, pca.n_components_, 1)
    print(">> PCA: number of components #{}".format(pca.n_components_))
    # printStat(pcs)
    evr = pca.explained_variance_ratio_
    # print(evr)
    evr_cumsum = np.cumsum(evr)
    # print(evr_cumsum)
    cut_evr = find_nearest(evr_cumsum, threshold)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
    df_pca = pd.DataFrame(data=pcs[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])
    loadings = pca.components_.T
    SSL = np.sum(loadings ** 2, axis=1)
    n_cols = 5
    if nPCs % n_cols == 0:
        n_rows = int(nPCs / n_cols)
    else:
        n_rows = nPCs // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (colname, ax) in enumerate(tqdm(zip(df_pca, axes.ravel()))):
        # print(i)
        component = df_pca[colname].values
        im = ax.imshow(component.reshape(regionshape), cmap='simple_list')
        ax.set_title("{}. component".format((i + 1)))
    fig.subplots_adjust(right=0.8)
    # divider = make_axes_locatable(axes)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, ax=ax)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.2)
    fig.colorbar(im, ax=axes, location='right', shrink=0.6)
    plt.show()
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

        plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
        plt.show()
        # Nc = 2
        # MaxPCs = nPCs# + 1
        # if MaxPCs % Nc != 0:
        #     Nr = int((nPCs + 1) / Nc)
        # else:
        #     Nr = int(nPCs/Nc)
        #  #MaxPCs
        # heights = [regionshape[1] for r in range(Nr)]
        # widths = [regionshape[0] for r in range(Nc)]
        # fig_width = 5.  # inches
        # fig_height = fig_width * sum(heights) / sum(widths)
        # fig, axs = plt.subplots(Nr, Nc, figsize=(fig_width, fig_height), dpi=600, constrained_layout=True,
        #                         gridspec_kw={'height_ratios': heights})
        # images = []
        # pc = 0
        # image = copy.deepcopy(pcs)
        # from sklearn.preprocessing import minmax_scale
        # image = minmax_scale(image.ravel(), feature_range=(10, 255)).reshape(image.shape)
        # for r in range(Nr):
        #     for c in range(Nc):
        #         # Generate data with a range that varies from one plot to the next.
        #         arrayPC = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
        #         for idx, coor in enumerate(localCoor):
        #             arrayPC[coor[0], coor[1]] = image[idx, pc]
        #         images.append(axs[r, c].imshow(arrayPC.T, origin='lower',
        #                                            cmap='simple_list'))  # 'RdBu_r')) #
        #         axs[r, c].label_outer()
        #         axs[r, c].set_axis_off()
        #         axs[r, c].set_title('PC{}'.format(pc + 1), fontsize=10, pad=0.25)
        #         fig.subplots_adjust(top=0.95, bottom=0.02, left=0,
        #                             right=1, hspace=0.14, wspace=0)
        #         pc += 1
        #
        # def update(changed_image):
        #     for im in images:
        #         if (changed_image.get_cmap() != im.get_cmap()
        #                 or changed_image.get_clim() != im.get_clim()):
        #             im.set_cmap(changed_image.get_cmap())
        #             im.set_clim(changed_image.get_clim())
        #             im.set_tight_layout('tight')
        #
        # for im in images:
        #     im.callbacks.connect('changed', update)
        # fig.suptitle("PC images {}:reg {}".format(filename, regID))
        # plt.show()
    # +------------------------------+
    # |   Agglomerating Clustering   |
    # +------------------------------+
    nCl = 4
    agg = AgglomerativeClustering(n_clusters=nCl)
    assignment = agg.fit_predict(df_pca.values)  # on pca
    # mglearn.discrete_scatter(np.array([i[0] for i in localCoor]),
    #                          np.array([i[1] for i in localCoor]), assignment, labels=np.unique(assignment))
    # plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
    # plt.title("Agglomerative Clustering")
    # plt.show()
    # seg = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
    # for idx, coor in enumerate(localCoor):
    #     seg[coor[0], coor[1]] = assignment[idx] + 1
    # plt.imshow(seg)
    plt.imshow(assignment.reshape(regionshape), cmap='simple_list')
    plt.title("Agglomerative segmentation")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 10), dpi=200)
    # plt.scatter(df_pca.PC_1, df_pca.PC_2, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
    # plt.scatter(df_pca.PC_1, df_pca.PC_2, c=assignment, facecolors='None', edgecolors=cm.tab20(0), alpha=0.5)
    mglearn.discrete_scatter(df_pca.PC_1, df_pca.PC_2, assignment, alpha=0.5)  # , labels=np.unique(assignment))
    plt.xlabel('PC1 ({}%)'.format(round(evr[0] * 100, 2)), fontsize=30)
    plt.ylabel('PC2 ({}%)'.format(round(evr[1] * 100, 2)), fontsize=30)
    plt.tick_params(size=10, color='black')
    # tick and axis theme
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['tissue {}'.format(c + 1) for c in range(nCl)], loc='upper right')
    ax = plt.gca()  # Get the current Axes instance
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)
    plt.suptitle("PCA performed with {} features".format(pca.n_features_), fontsize=30)
    plt.show()
    # +-------------------+
    # |   HC_clustering   |
    # +-------------------+
    # pca = PCA(random_state=RandomState, n_components=nPCs)
    # pcs = pca.fit_transform(pixel_features)
    # HC_method = 'ward'
    # HC_metric = 'euclidean'
    # Y = sch.linkage(mz_features, method=HC_method, metric=HC_metric)
    # Z = sch.dendrogram(Y, orientation='left') #, color_threshold=thre_dist
    # if __name__ == '__main__':
    HC_method = 'ward'
    HC_metric = 'euclidean'
    # plot dendogram
    fig = plt.figure(figsize=(15, 15), dpi=300)
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    Y = sch.linkage(mz_features, method=HC_method, metric=HC_metric)
    thre_dist = 78  # TODO: how to fix it?
    Z = sch.dendrogram(Y, color_threshold=thre_dist, orientation='left') #
    fig.gca().invert_yaxis()
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    # plot matrix/feature(sorted)
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8]) #left, bottom, width, height
    HC_idx = np.array(Z['leaves'])
    mz_features_sorted = mz_features[HC_idx] #mz_features[HC_idx] #df_pca.values[HC_idx]
    im = axmatrix.matshow(mz_features_sorted, aspect='auto', origin='lower', cmap=cm.YlGnBu, interpolation='none')
    fig.gca().invert_yaxis()
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # colorbar
    axcolor = fig.add_axes([0.92, 0.1, 0.02, 0.80])
    axcolor.tick_params(labelsize=10)
    plt.colorbar(im, cax=axcolor)
    fig.suptitle('hierarchical clustering of ion images \n method: {}, metric: {}, threshold: {}'.format(
        HC_method, HC_metric, thre_dist), fontsize=16)
    fig.show()

    HC_labels = sch.fcluster(Y, thre_dist, criterion='distance')
    # # prepare label data
    elements, counts = np.unique(HC_labels, return_counts=True)
    print(elements, counts)
    # sarray1 = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
    # for idx, coor in enumerate(localCoor):
    #     sarray1[coor[0], coor[1]] = HC_labels[idx]
    #
    # fig, ax = plt.subplots(figsize=(6, 8))
    # sarrayIm = ax.imshow(sarray1)
    # fig.colorbar(sarrayIm)
    # ax.set_title('reg{}: HC Clustering'.format(regID), fontsize=15, loc='center')
    # plt.show()

    # +----------------+
    # |   clustering   |
    # +----------------+
    img_std = pixel_features.T.reshape((pixel_features.shape[1], regionshape[0], regionshape[1]))
    mean_imgs = []
    total_SSLs = []
    for label in elements:
        idx = np.where(HC_labels == label)[0]
        total_SSL = np.sum(SSL[idx])
        current_cluster = img_std[idx]
        mean_img = np.mean(current_cluster, axis=0)
        total_SSLs.append(total_SSL)
        mean_imgs.append(mean_img)

    rank_idx = np.argsort(total_SSLs)
    # plot together
    fig, axes = plt.subplots(4, 13, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
    for i, ax in tqdm(zip(range(elements.shape[0]), axes.ravel())):
        # print(i)
        idx = rank_idx[i]
        total_SSL = total_SSLs[idx]
        count = counts[idx]
        mean_img = mean_imgs[idx]
        im = ax.imshow(mean_img, cmap='simple_list')
        ax.set_title("Rank #{} # of images: {}".format(i, count))  # total_SSL: {} , round(total_SSL, 4)
    fig.subplots_adjust(right=0.8)
    # divider = make_axes_locatable(axes)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, ax=ax)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.2)
    # fig.colorbar(im, ax=axes, location='right', shrink=0.6)
    plt.show()
    # +------------------+
    # |      UMAP        |
    # +------------------+
    u_neigh = 12    # from grid-parameter search in Hang Hu paper(visual plot)
    u_comp = 3
    u_min_dist = 0.025  # from grid-param Hang Hu paper(visual plot)
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
    data_umap = reducer.fit_transform(df_pca.values) # .iloc[:, 0:nPCs] or df_pca.values)  # on pca
    df_pca_umap = copy.deepcopy(df_pca)
    for i in range(reducer.n_components):
        df_pca_umap.insert(df_pca_umap.shape[1], column='umap_{}'.format(i + 1), value=data_umap[:, i])
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

    df_pca_umap_hdbscan = copy.deepcopy(df_pca_umap)
    df_pca_umap_hdbscan.insert(df_pca_umap_hdbscan.shape[1], column='hdbscan_labels', value=labels)

    # +--------------------------------------+
    # |   Segmentation on PCA+UMAP+HDBSCAN   |
    # +--------------------------------------+
    imdata = labels.reshape(regionshape) #np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
    imdata = nnPixelCorrect(imdata, 19, 3)
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(imdata, cmap='simple_list')
    fig.colorbar(im)
    ax.set_title('reg{}: HDBSCAN labeling'.format(regID), fontsize=15, loc='center')
    plt.show()
    if save_rseg:
        namepy = os.path.join(regDir, '{}_hdbscan-label.npy'.format(exprun))
        np.save(namepy, imdata)

    # +-----------------+
    # |       GMM       |
    # +-----------------+
    n_components = 5
    span = 5
    n_component = _generate_nComponentList(n_components, span)
    repeat = 2
    print("Consider for gmm >>", df_pca.columns[0:nPCs])
    for i in range(repeat):  # may repeat several times
        for j in range(n_component.shape[0]):  # ensemble with different n_component value
            StaTime = time.time()
            gmm = GMM(n_components=n_component[j], max_iter=5000)  # max_iter does matter, no random seed assigned
            # labels = gmm.fit_predict(df_pca_umap.iloc[:, 0:nPCs+reducer.n_components])     #todo data_umap
            labels = gmm.fit_predict(df_pca.values)
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
    if save_rseg:
        savecsv = os.path.join(regDir, '{}.csv'.format(exprun))
        df_pca_umap_hdbscan_gmm.to_csv(savecsv, index=False, sep=',')

    nGs = retrace_columns(df_pca_umap_hdbscan_gmm.columns.values, 'gmm')
    df_gmm_labels = df_pca_umap_hdbscan_gmm.iloc[:, -nGs:]
    # print("gmm label: ", nGs)

    fig, axes = plt.subplots(2, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
    for (columnName, columnData), ax in zip(df_gmm_labels.iteritems(), axes.ravel()):
        print('Column Name : ', columnName)
        print('Column Contents : ', np.unique(columnData.values))
        imdata = columnData.values.reshape(regionshape)
        im = ax.imshow(imdata, cmap='simple_list')
        fig.colorbar(im, ax=ax, shrink=0.5)
        ax.set_title('reg{}: umap_{}'.format(regID, columnName), fontsize=15, loc='center')
        # plt.show()
        if save_rseg:
            namepy = os.path.join(regDir, '{}_{}.npy'.format(exprun, columnName))
            np.save(namepy, imdata)
    fig.show()
    return

def msmlfunc6(mspath, regID, exprun): # exprun,
    """
    performs ML on resampled and peak picked spectra based on tolerance
    without interpolation...
    """
    colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
    color_bin = 100
    mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_list', colors=colors, N=color_bin))
    umap_hdbscan_ = False
    kde_cluster_ = False
    plot_pca = True
    RandomState = 20210131
    save_rseg = False
    # +------------------------------------+
    # |     read data and save region      |
    # +------------------------------------+
    dirname = os.path.dirname(mspath)
    basename = os.path.basename(mspath)
    filename, ext = os.path.splitext(basename)
    regDir = os.path.join(dirname, 'reg_{}'.format(regID))
    if not os.path.isdir(regDir):
        os.mkdir(regDir)
    # peakfilename = os.path.join(regDir, 'peak_picked_reg_{}.h5'.format(regID)) #'realigned_15000_20_100.h5')#
    peakfilename = os.path.join(regDir,'peak_picked_reg_1.h5')# 'peakspectra_step01_snr7_int1000.h5') # peakspectra_centroid_step01_snr7_int1000.h5')
    # regname = os.path.join(regDir, '{}_reg_{}_{}.h5'.format(filename, regID))
    ImzObj = ImzmlAll(mspath)
    if os.path.isfile(peakfilename):
        with h5py.File(peakfilename, 'r') as pfile:
            peakspectra = np.array(pfile.get('peakspectra'))
            # ionimages = np.array(pfile.get('ionimages'))
            peakmzs = np.array(pfile.get('peakmzs'))
            # regionshape = np.array(pfile.get('regionshape'))
            # localCoords = list(pfile.get('coordinates'))
            regionshape, localCoords = np.array(ImzObj.get_region_shape_coords(regID=regID))
    else:
        spectra, refmz, regionshape, localCoords = ImzObj.resample_region(regID, tol=0.01, savedata=True)
        # print(regionshape, type(regionshape))
        # spectra_smoothed = ImzObj.smooth_spectra(spectra, window_length=9, polyorder=2)
        peakspectra, peakmzs = ImzObj.peak_pick(spectra, refmz)
        with h5py.File(peakfilename, 'w') as pfile: # saves the data
            pfile['peakspectra'] = peakspectra
            pfile['peakmzs'] = peakmzs
            pfile['regionshape'] = np.array(regionshape)
            pfile['coordinates'] = localCoords
    savecsv = os.path.join(regDir, 'results_{}.csv'.format(exprun))
    # +------------------------------------+
    # |    get spectra from ion images     |
    # +------------------------------------+
    # ionimages = ionimages.transpose(1, 2, 0)
    # ionimages_flat = ionimages.reshape(-1, ionimages.shape[2])
    # del_idx = np.where(np.mean(ionimages_flat, axis=1) == 0)[0]
    # peakspectra = np.delete(ionimages_flat, del_idx, axis=0)    # foreground x nm/z or npeak
    peakspectra_tic = np.zeros_like(peakspectra)
    for s in range(peakspectra.shape[0]):
        peakspectra_tic[s, :] = normalize_spectrum(peakspectra[s, :], normalize='tic')
    # +----------------------------+
    # |       UMAP-HDBSCAN         |
    # +----------------------------+
    if umap_hdbscan_:
        peakspectra_tic_ss = SS().fit_transform(peakspectra_tic)
        data_umap = umap_it(peakspectra_tic_ss)
        u_comp = data_umap.shape[1]
        df_umap = pd.DataFrame(data=data_umap[:, 0:u_comp], columns=['umap_%d' % (i + 1) for i in range(u_comp)])
        df_umap.to_csv(savecsv, index=False, sep=',')

        labels = hdbscan_it(data_umap)
        print("UMAP-HDBSCAN found {} labels in the data".format(np.unique(labels)))
        df_umap_hdbscan = copy.deepcopy(df_umap)
        df_umap_hdbscan.insert(df_umap_hdbscan.shape[1], column='hdbscan_labels', value=labels)
        # savecsv = os.path.join(regDir, 'umap_hdbscan_{}.csv'.format(exprun))
        df_umap_hdbscan.to_csv(savecsv, index=False, sep=',')

    # +------------------------------------+
    # |       remove sparse features       |
    # +------------------------------------+
    nz_cent = []
    for feat in range(peakspectra_tic.shape[1]):
        div = round((100 * (len(peakspectra_tic[:, feat].nonzero()[0]) / peakspectra_tic.shape[0])), 2)
        nz_cent.append(div)
    nz_cent = np.array(nz_cent, dtype=np.float32)
    remove_perc_ = 10 #25

    fig, ax = plt.subplots(dpi=200)
    offsetbox = TextArea("{} out of {} features has less than {}% \nnon-zero elements: too sparse".format(
        len(np.where(nz_cent <= remove_perc_)[0]), len(nz_cent), remove_perc_))
    xy = (2, 2000)
    ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.2, xy[1]),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="->"))
    ax.hist(nz_cent, bins=50, color='blue')
    ax.set_xlabel("%", fontsize=10)
    ax.set_ylabel("#features", fontsize=10)
    ax.set_title("% of non-zero elements in features", fontsize=10, fontweight='bold')
    ax.add_artist(ab)
    plt.show()
    perc_indexing = np.argsort(nz_cent)[::-1]  # high to low indexing
    nz_mz_list = list(map(lambda nz, mz: (round(nz, 4), round(mz, 4)), nz_cent[perc_indexing], peakmzs[perc_indexing]))
    ionimages = _2d_to_3d(peakspectra, regionshape, localCoords)
    masterPlot(ionimages[..., perc_indexing[0:50]], nz_mz_list[0:50], Nc=5,
               Title="dense ion images")
    masterPlot(ionimages[..., perc_indexing[-50:][::-1]], nz_mz_list[-50:][::-1], Nc=5,
               Title="sparse ion images")
    peakspec_tic_dense = np.delete(peakspectra_tic, np.where(nz_cent <= remove_perc_)[0], axis=1)
    peakspec_tic_sparse = np.delete(peakspectra_tic, np.where(nz_cent > remove_perc_)[0], axis=1)
    peakmzs_dense = np.delete(peakmzs, np.where(nz_cent <= remove_perc_)[0]) #.reshape(-1, 1)
    peakmzs_sparse = np.delete(peakmzs, np.where(nz_cent > remove_perc_)[0]) #.reshape(-1, 1)
    # ImzObj.get_ion_images(regID, peakspec_dense, peakmzs_dense, top=False)
    # ImzObj.get_ion_images(regID, peakspec_sparse, peakmzs_sparse, top=False)
    images_dense = _2d_to_3d(peakspec_tic_dense, regionshape, localCoords)
    images_sparse = _2d_to_3d(peakspec_tic_sparse, regionshape, localCoords)
    # save sparse images for visualization:
    # saveimages(images_sparse, list(np.squeeze(peakmzs_sparse)), os.path.join(regDir, 'ionimages_sparse_{}pc'.format(remove_perc_)))
    # images_dense_flat = images_dense.reshape(-1, images_dense.shape[2])
    # images_dense_flat_norm = np.zeros_like(images_dense_flat)
    # for s in range(images_dense_flat.shape[0]):
    #     images_dense_flat_norm[s, :] = normalize_spectrum(images_dense_flat[s, :], normalize='tic')
    # images_dense_flat_norm_ss = SS().fit_transform(images_dense_flat_norm)
    # images_dense_norm = images_dense_flat_norm.reshape(regionshape[0], regionshape[1], images_dense_flat_norm.shape[1]) # ss
    # images_feat_flat_norm_pt = PT(method="yeo-johnson").fit_transform(images_feat_flat_norm)
    # +------------------------------+
    # |  cluster peak M/Zs with KDE  |
    # +------------------------------+
    # if kde_cluster_:
    bw = 30
    nClusters, centroids = kdemzs(peakmzs_dense.reshape(-1, 1), bw)

    # +--------------------------------+
    # |   ion-image/feature selection  |
    # +--------------------------------+
    # class PFA(object):
    #     def __init__(self, n_features, q=None):
    #         self.q = q
    #         self.n_features = n_features
    #
    #     def fit(self, X):
    #         if not self.q:
    #             self.q = X.shape[1]
    #
    #         # sc = StandardScaler()
    #         X = SS().fit_transform(X)
    #
    #         pca = PCA(n_components=self.q).fit(X)
    #         A_q = pca.components_.T
    #
    #         kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
    #         clusters = kmeans.predict(A_q)
    #         cluster_centers = kmeans.cluster_centers_
    #
    #         u_labels = np.unique(clusters)
    #         for i in u_labels:
    #             plt.scatter(A_q[clusters == i, 0], A_q[clusters == i, 1], label=i)
    #         plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=10, color='k')
    #         plt.legend()
    #         plt.show()
    #
    #         # labels = hdbscan_it(A_q)
    #
    #         dists = defaultdict(list)
    #         for i, c in enumerate(clusters):
    #             dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
    #             dists[c].append((i, dist))
    #
    #         self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
    #         self.features_ = X[:, self.indices_]
    #
    # pfa = PFA(n_features=100)
    # pfa.fit(images_feat_flat_norm_ss)
    # X = pfa.features_
    # column_indices = pfa.indices_
    # variance explained threshold

    # +------------------------------------+
    # |        VE threshold with SVD       |
    # +------------------------------------+
    s1_v = []
    s2_v = []
    for m in range(images_dense.shape[2]):
        image = images_dense[..., m]
        u, s, vT = svd(image, full_matrices=False)
        var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=3)
        s1_v.append(var_explained[0])
        s2_v.append(sum(var_explained[0:2]))
    s1_v = np.array(s1_v)
    s2_v = np.array(s2_v)
    var_thre = round(np.percentile(s2_v, 25), 5)  # 0.25
    print("25 percentile value of ve: ", var_thre)
    fig, ax = plt.subplots(dpi=300)
    ax.vlines(peakmzs_dense, ymin=0, ymax=s1_v, colors='r', label='s1_v')
    ax.vlines(peakmzs_dense, ymin=0, ymax=s2_v, colors='r', alpha=0.2, label='(s1+s2)_v')
    ax.hlines(var_thre, xmin=peakmzs_dense[0] - 5, xmax=peakmzs_dense[-1] + 5, colors='black',
              label='25 percentile: {}'.format(var_thre))
    ax.legend(loc='upper right')
    ax.set_title("Variance explained by SVD components", fontsize=15)
    ax.set_xlabel("mz features/channels", fontsize=10)
    ax.set_ylabel("variance", fontsize=10)
    plt.show()

    print("number of ion images exceed ve threshold: ", len(np.where(s2_v >= var_thre)[0]))
    peakmzs_dense_ve = peakmzs_dense[np.where(s2_v >= var_thre)[0]].reshape(-1)
    s2_indexing = np.argsort(s2_v)[::-1]
    # s2_sorted = s2_v[s2_sorted_idx]
    # s2_25_idx = s2_sorted_idx[np.where(s2_sorted > var_thre)]
    # s2_25_idx_drop = s2_sorted_idx[np.where(s2_sorted <= var_thre)]
    # print(len(s2_25_idx), s2_25_idx)
    # visualize the sorted images
    ve_mz_list = list(map(lambda ve, mz: (round(ve, 4), round(mz, 4)), s2_v[s2_indexing], peakmzs_dense[s2_indexing]))
    # savedir = os.path.join(regDir, 'ion_images_ve_sorted_{}'.format(var_thre))
    masterPlot(images_dense[..., s2_indexing[0:50]], ve_mz_list[0:50], Nc=5, Title="ve_exceed")
    # saveimages(images_dense_norm_ss[..., s2_25_idx], ve_mz_list, savedir)
    # ve_mz_list = list(map(lambda ve, mz: (round(ve, 4), mz[0]), s2_v[s2_25_idx_drop], peakmzs_dense[s2_25_idx_drop]))
    savedir = os.path.join(regDir, 'dense_ve_dropped_{}'.format(var_thre))
    masterPlot(images_dense[..., s2_indexing[-50:][::-1]], ve_mz_list[-50:][::-1], Nc=5, Title="ve_under")
    # saveimages(images_dense_norm_ss[..., s2_25_idx_drop], ve_mz_list, savedir)
    # ImzObj = ImzmlAll(mspath)
    # ImzObj.get_ion_images(regID=regID, peakspectra=peakspec_dense[:, s1_25_idx][:, 0:50],
    #                       peakmzs=peakmzs_dense[s1_25_idx][:, 0:50])
    # ImzObj.get_ion_images(regID=regID, peakspectra=peakspec_dense[:, s1_25_idx][:, -50:],
    #                       peakmzs=peakmzs_dense[s1_25_idx][:, -50:])

    # the features are sorted or not, doesn't matter to PCA.
    # images_dense_flat_norm_ss_ve = images_dense_flat_norm_ss[:, np.where(s2_v > var_thre)[0]]
    # images_dense_norm_ss_ve = images_dense_flat_norm_ss_ve.reshape(regionshape[0], regionshape[1],
    #                                         images_dense_flat_norm_ss_ve.shape[1]) #.transpose(2, 0, 1)
    images_dense_ve = images_dense[..., np.where(s2_v >= var_thre)[0]]
    # +--------------------------------+
    # |    spatial coherence/chaos     |
    # +--------------------------------+
    from Esmraldi.esmraldi.segmentation import find_similar_images_spatial_coherence
    # images_dense_norm_ss_ve_sc, sc_idx = find_similar_images_spatial_coherence(images_dense_norm_ss_ve, 0, [60, 70, 80, 90])
    from Esmraldi.esmraldi.segmentation import spatial_coherence, spatial_chaos, average_distance_graph
    # from skimage.filters import sobel
    # image3D_drop = images_dense_norm_ss[..., s2_25_idx_drop]
    # image3D_kept = images_dense_norm_ss[..., s2_25_idx]
    total_coherence = []
    quantiles = [90] #, 70, 80, 90]
    upper = 100
    # for image3D in [image3D_drop, image3D_kept]:
    for quantile in quantiles:
        coherence_values = []
        for i in range(images_dense_ve.shape[-1]):
            image = images_dense_ve[..., i]
            imagenorm = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
            upper_threshold = np.percentile(imagenorm, upper)
            # edges = sobel(imagenorm)
            threshold = int(np.percentile(imagenorm, quantile))
            sc = spatial_coherence((imagenorm > threshold) & (imagenorm <= upper_threshold))
            coherence_values.append(sc)
        total_coherence.append(coherence_values)
    coherence_values = np.array(coherence_values)
    sc_thre = np.percentile(coherence_values, 10)
    boxplotit(data=total_coherence, labels=['qt-{}'.format(q) for q in quantiles], Title='Spatial coherence')
    # boxplotit(total_chaos, ['ve -25', 've +25'], 'Spatial chaos 70qt')
    # chaos_values = np.array(chaos_values)
    coherence_indexing = np.argsort(coherence_values)[::-1]
    # coherence_sorted = coherence_values[coherence_sorted_idx]
    # coherence_10_idx = coherence_sorted_idx[np.where(coherence_sorted > sc_thre)]
    # coherence_10_idx_ = coherence_sorted_idx[np.where(coherence_sorted <= sc_thre)]

    # images_dense_norm_ss_ve_sc_ = images_dense_norm_ss_ve[..., coherence_10_idx_]
    # peakmzs_dense_ve_sc = peakmzs_dense_ve[coherence_10_idx] #np.where(coherence_values >= sc_thre)[0]]
    # peakmzs_dense_ve_sc_ = peakmzs_dense_ve[coherence_10_idx_]
    sc_mz_list = list(map(lambda sc, mz: (round(sc, 4), round(mz, 4)), coherence_values[coherence_indexing], peakmzs_dense_ve[coherence_indexing]))
    masterPlot(images_dense_ve[..., coherence_indexing[0:50]], sc_mz_list[0:50], Nc=5, Title="Spatially coherent(best)")
    masterPlot(images_dense_ve[..., coherence_indexing[-50:][::-1]], sc_mz_list[-50:][::-1], Nc=5,
               Title="Spatially coherent(least)")
    images_dense_ve_sc = images_dense_ve[..., np.where(coherence_values >= sc_thre)[0]]
    peakmzs_dense_ve_sc = peakmzs_dense_ve[np.where(coherence_values >= sc_thre)[0]].reshape(-1)
    # chaos_sorted_idx = chaos_values.argsort()  # chaos is less and reverse than coherence
    # images_kept_coherence_sorted = image3D_kept[..., coherence_sorted_idx].transpose(2, 0, 1)
    # images_kept_chaos_sorted = image3D_kept[..., chaos_sorted_idx].transpose(2, 0, 1)
    # savedir = os.path.join(regDir, 'coherent_ion_images'.format(sc_thre))
    # sc_mz_list = list(map(lambda sc, mz: (round(sc, 4), mz), coherence_values[coherence_10_idx], peakmzs_dense_ve_sc))
    # saveimages(images_dense_norm_ss_ve_sc, sc_mz_list, savedir)
    # savedir_ = os.path.join(regDir, 'incoherent_{}'.format(sc_thre))
    # images_dense_norm_ss_ve_sc_ = images_dense_norm_ss_ve[..., coherence_10_idx_]
    # peakmzs_dense_ve_sc_ = peakmzs_dense_ve[coherence_10_idx_]
    # sc_mz_list_ = list(
    #     map(lambda sc, mz: (round(sc, 4), mz), coherence_values[coherence_10_idx_], peakmzs_dense_ve_sc_))
    # saveimages(images_dense_norm_ss_ve[..., coherence_10_idx_], sc_mz_list_, savedir_)

    # +------------+
    # |    PCA     |
    # +------------+
    images_dense_flat_ve_sc = images_dense_ve_sc.reshape(-1, images_dense_ve_sc.shape[-1])
    images_dense_flat_ve_sc_ss = SS().fit_transform(images_dense_flat_ve_sc)
    pca = PCA(random_state=RandomState)  #n_components=100,
    pcs = pca.fit_transform(images_dense_flat_ve_sc_ss) # includes both foreground and background
    pca_range = np.arange(1, pca.n_components_, 1)
    print(">> PCA: number of components #{}".format(pca.n_components_))
    evr = pca.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)
    threshold = 0.85
    cut_evr = find_nearest(evr_cumsum, threshold)
    nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
    print(">> Nearest variance to threshold {:.4f} explained by #PCA components {}".format(cut_evr, nPCs))
    if nPCs >= 50:  # 2 conditions to choose nPCs.
        nPCs = 50
    df_pca = pd.DataFrame(data=pcs[:, 0:nPCs], columns=['PC_%d' % (i + 1) for i in range(nPCs)])

    if plot_pca:
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

    images_pca = df_pca.values #np.transpose(df_pca.values, (1, 0))
    images_pca = images_pca.reshape(regionshape[0], regionshape[1], images_pca.shape[-1])
    componentList = list(df_pca.columns)
    masterPlot(images_pca, componentList, Title='PCA components', Nc=4)
    # +-----------------+
    # |       GMM       |
    # +-----------------+
    n_components = 5
    span = 5
    n_component = _generate_nComponentList(n_components, span)
    repeat = 1
    print("Consider for gmm >>", df_pca.columns[0:nPCs])
    df_gmm_label = pd.DataFrame()
    for i in range(repeat):  # may repeat several times
        for j in range(n_component.shape[0]):  # ensemble with different n_component value
            StaTime = time.time()
            gmm = GMM(n_components=n_component[j], max_iter=5000,
                      random_state=RandomState)  # max_iter does matter, no random seed assigned
            labels = gmm.fit_predict(df_pca.values)
            index = j + 1 + i * n_component.shape[0]
            title = 'gmm_' + str(index) + '_' + str(n_component[j]) + '_' + str(i)
            df_gmm_label[title] = labels
            SpenTime = (time.time() - StaTime)
            # progressbar
            print('{}/{}, finish classifying {}, running time is: {} s'.format(index, repeat * span, title,
                                                                               round(SpenTime, 2)))
    def relabel(df_pixel_label):
        pixel_relabel = np.empty([df_pixel_label.shape[0], 0])
        for i in range(df_pixel_label.shape[1]):
            column = df_pixel_label.iloc[:, i].value_counts()  # counts number of individual labels in each Gaussian dist
            labels_old = column.index.values.astype(int)  # kinda take unique labels
            labels_new = np.linspace(0, labels_old.shape[0] - 1, labels_old.shape[0]).astype(int)
            column_new = df_pixel_label.iloc[:, i].replace(labels_old, labels_new)  # pd.series
            pixel_relabel = np.append(pixel_relabel, column_new.values.astype(int).reshape(df_pixel_label.shape[0], 1),
                                      axis=1)
        return pixel_relabel

    gmm_label_relab = relabel(df_gmm_label)
    df_gmm_label_relab = pd.DataFrame(gmm_label_relab, columns=df_gmm_label.columns)
    Nr = 5
    Nc = 1
    heights = [regionshape[1] for r in range(Nr)]
    widths = [regionshape[0] for r in range(Nc)]
    fig_width = 5.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    fig, axes = plt.subplots(Nr, Nc, dpi=400, figsize=(fig_width, fig_height),
                             # constrained_layout=True,
                             subplot_kw={'xticks': (), 'yticks': ()})  #
    for (columnName, columnData), ax in zip(df_gmm_label_relab.iteritems(), axes.ravel()):
        print('Column Name : ', columnName)
        print('Column Contents : ', np.unique(columnData.values))
        imdata = columnData.values.reshape(regionshape)
        im = ax.imshow(imdata.T, origin='lower', cmap='msml_list')
        ax.set_title('reg{}: {}'.format(regID, columnName), fontsize=15, loc='center')
        plt.colorbar(im, ax=ax, shrink=0.5)
    plt.show()

    # +--------------------------------------+
    # |    ion image/ feature clustering     |
    # +--------------------------------------+
    # take foreground spectra(on-tissue?) -> TIC normalized
    peakspec_tic_dense_ve = peakspec_tic_dense[:, np.where(s2_v >= var_thre)[0]]
    peakspec_tic_dense_ve_sc = peakspec_tic_dense_ve[:, np.where(coherence_values >= sc_thre)[0]]
    # peakspec_dense_ve_sc_tic = np.zeros_like(peakspec_dense_ve_sc)
    # for s in range(peakspec_dense_ve.shape[0]):
    #     peakspec_dense_ve_sc_tic[s, :] = normalize_spectrum(peakspec_dense_ve_sc[s, :], normalize='tic')
    # take 99th percentile and clip intensities above that/ winsorize every image channel
    wins_perc = 0.95
    scale_up = round(peakspec_tic_dense_ve_sc.shape[0]*wins_perc)
    peakspec_tic_dense_ve_sc_wins = np.zeros_like(peakspec_tic_dense_ve_sc)
    for m in range(peakspec_tic_dense_ve_sc.shape[1]):
        pixels = peakspec_tic_dense_ve_sc[:, m]
        thre = pixels[np.argsort(pixels)[scale_up]]
        pixels[pixels > thre] = thre
        pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))
        peakspec_tic_dense_ve_sc_wins[:, m] = pixels
    # normalize min-max scale

    # Transpose to get feature matrix
    features = peakspec_tic_dense_ve_sc_wins.T
    # standardize pixel(foreground+background) features(probably already done)
    # PCA -> pixel_feature_std
    pca = PCA(random_state=RandomState, n_components=nPCs)  # pca object n_components=100,
    pcs = pca.fit_transform(images_dense_flat_ve_sc_ss)
    loadings = pca.components_.T
    SSL = np.sum(loadings ** 2, axis=1)
    # HC -> feature matrix
    HC_method = 'ward'
    HC_metric = 'euclidean'
    Y = sch.linkage(features, method=HC_method, metric=HC_metric)
    Z = sch.dendrogram(Y, no_plot=True)
    HC_idx = Z['leaves']
    HC_idx = np.array(HC_idx)
    plt.figure(figsize=(15, 10))
    thre_dist = 50#78
    Z = sch.dendrogram(Y, color_threshold=thre_dist)
    plt.title('hierarchical clustering of ion images \n method: {}, metric: {}, threshold: {}'.format(
        HC_method, HC_metric, thre_dist))
    plt.show()
    # sort feature matrix based on HC results
    # How much PCA loading covered by each cluster
    # rank them based on SSL?
    features_sorted = features[HC_idx]
    fig = plt.figure(figsize=(15, 15), dpi=300)
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    Z = sch.dendrogram(Y, color_threshold=thre_dist, orientation='left')
    fig.gca().invert_yaxis()
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    # fig.show()
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    im = axmatrix.matshow(features_sorted, aspect='auto', origin='lower', cmap=cm.YlGnBu, interpolation='none')
    fig.gca().invert_yaxis()
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axcolor = fig.add_axes([0.92, 0.1, 0.02, 0.80])
    axcolor.tick_params(labelsize=10)
    plt.colorbar(im, cax=axcolor)
    fig.suptitle('hierarchical clustering of ion images \n method: {}, metric: {}, threshold: {}'.format(
                    HC_method, HC_metric, thre_dist), fontsize=16)
    plt.show()

    HC_labels = sch.fcluster(Y, thre_dist, criterion='distance')
    elements, counts = np.unique(HC_labels, return_counts=True)
    images_dense_ve_sc_ss = images_dense_flat_ve_sc_ss.reshape(regionshape[0], regionshape[1],
                                            images_dense_flat_ve_sc_ss.shape[-1]) #.transpose(2, 0, 1)

    mean_imgs = []
    total_SSLs = []
    for label in elements:
        idx = np.where(HC_labels == label)[0]
        # total SSL
        total_SSL = np.sum(SSL[idx])
        # imgs in the cluster
        current_cluster = images_dense_ve_sc_ss.transpose(2, 0, 1)[idx]
        # savedir = os.path.join(regDir, 'cluster_{}'.format(label))
        # saveimages(np.transpose(current_cluster, (1, 2, 0)), peakmzs_dense_ve[idx], savedir)
        # average img
        mean_img = np.mean(current_cluster, axis=0)
        # accumulate data
        total_SSLs.append(total_SSL)
        mean_imgs.append(mean_img)

    rank_idx = np.argsort(total_SSLs)

    mean_imgs_ = [mean_imgs[r] for r in rank_idx]
    total_SSLs_ = [total_SSLs[r] for r in rank_idx]
    counts_ = [counts[r] for r in rank_idx]
    vList = ['Cluster-{}:SSL {} count {}'.format(c+1, round(i, 4), j) for c, (i, j) in
             enumerate(zip(total_SSLs_, counts_))]
    masterPlot(np.transpose(mean_imgs_, (1, 2, 0)), vList, Title='Mean ion images of clusters with SSL rank')


    def get_colors(inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))

    colors = get_colors(elements, mtl.colormaps['msml_list'])  # mtl.colormaps['RdYlBu_r']) #
    # print(mtl.colormaps['msml_list'])
    print(colors)
    Nr, Nc = len(elements), 1
    heights = [5 for r in range(Nr)]
    widths = [15 for c in range(Nc)]
    fig_width = 5.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    fig, axes = plt.subplots(Nr, Nc, figsize=(fig_width, fig_height), dpi=400,
                             constrained_layout=True,
                             gridspec_kw={'height_ratios': heights})  # , sharex=True)
    peak_mean = np.mean(peakspec_tic_dense_ve_sc, axis=0)
    for i, ax in enumerate(axes.ravel()):
        peak_norm = (peak_mean - np.min(peak_mean)) / np.ptp(peak_mean)
        label_idx_rest = np.where(HC_labels != i + 1)
        peak_norm[label_idx_rest] = 0
        ax.vlines(peakmzs_dense_ve_sc, ymin=0, ymax=peak_norm, colors=colors[i], label=i + 1)
        ax.legend()

    fig.suptitle("Mean spectrum of clusters", y=0.999)
    fig.text(0.5, 0.003, 'm/z', ha='center')
    fig.text(0.01, 0.5, 'intensity(normalized)', va='center', rotation='vertical')
    plt.show()

    # finding pca outliers
    pca = PCA(random_state=RandomState)  # pca object n_components=100,
    pcs = pca.fit_transform(images_dense_flat_ve_sc_ss)
    loadings_all = pca.components_.T
    AL = abs(loadings_all)  # .T brings it into feature space from component space
    AL_max_indexing = np.argmax(AL, axis=0)     # index of maximum absolute loadings
    # AL_max_indexing matches with topfeat of pca2
    mask = np.in1d(np.arange(0, len(AL_max_indexing)), np.unique(AL_max_indexing), invert=True)
    feature_missing = np.arange(0, len(AL_max_indexing))[mask]
    displayImage(np.mean(images_dense_ve_sc[..., feature_missing], axis=2))

    SAL = np.sum(AL, axis=1)
    SAL_indexing = np.argsort(SAL)[::-1]

    SL = loadings_all**2
    SSL_all = np.sum(SL, axis=1)
    SSL_indexing = np.argsort(SSL_all)[::-1]
    # print("Corr", np.corrcoef(images_dense_flat_norm_ss_ve[:, 729], images_dense_flat_norm_ss_ve[:, 892])[1,0]) # pearsonr

    from pca import pca as pca2
    model = pca2(random_state=RandomState)
    results = model.fit_transform(images_dense_flat_ve_sc_ss, verbose=False)
    print(results['topfeat'])
    best_feat_idx = [int(i) - 1 for i in results['topfeat']['feature'].values]
    weak_feat_idx = np.where(results['topfeat']['type'].values == 'weak')[0] - 1
    max_AL = [] # maximum abs loading means best feature that influences a PC
    for pc in range(AL.shape[1]):
        max_AL.append(np.where(AL[:, pc] == max(AL[:, pc]))[0][0])

    # masterPlot(images_dense_norm_ss_ve[max_AL[0:20], ...], peakmzs_dense_ve[max_AL[0:20]], Nc=4, Title='PCA best features')
    # save images in drive
    min_AL = list(reversed(max_AL)) # weak features that influences a PC.
    # masterPlot(images_dense_norm_ss_ve[min_AL[0:20], ...], peakmzs_dense_ve[min_AL[0:20]], Nc=4, Title='PCA weak features')

    from skimage.filters import threshold_multiotsu
    n_classes = 5
    img_thresholds = np.empty([n_classes - 1, 0])
    img_segs = np.empty([0, regionshape[0], regionshape[1]])
    print(img_segs.shape)
    for idx in min_AL[:20]:
        # for i in index:
        image = images_dense_ve_sc.transpose(2,0,1)[idx]
        thresholds = threshold_multiotsu(image, classes=n_classes)  # first 2 columns are spatial index
        img_seg = np.digitize(image, bins=thresholds)
        img_thresholds = np.append(img_thresholds, thresholds.reshape(thresholds.shape[0], 1), axis=1)
        img_segs = np.append(img_segs, img_seg.reshape(1, img_seg.shape[0], img_seg.shape[1]), axis=0)
    masterPlot(imageArray=np.transpose(img_segs, (1, 2, 0)), valueList=min_AL[:20], Nc=4, Title='min AL')
    print("Here")
    return

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet=None, level=2):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    num_ = 1.0   # raising this increases spikes
    den_ = 0.6745*4
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
    bestwvlt: e.g.: 'db8' or 'rbio2.8' if not given finds.
    smooth_array: if True returns smoothed spectra array with best wavelet.
    plot_fig: plots figure if True.
    e.g.: bestWvltForRegion(spec_data1, bestWvlt='db8', smoothed_array=True, plot_fig=True)
    """
    if bestWvlt is None:
        wvList = []
        smList = []
        for nS in tqdm(range(spectra_array.shape[0]), desc='#spectra%'):
            signal = spectra_array[nS, :]   #   copy.deepcopy(spectra_array[nS, :])
            _, wv, sm = waveletPickwSim(signal, plot_fig=False, verbose=False)
            wvList.append(wv)
            smList.append(sm)
        if plot_fig:
            keys, counts = np.unique(wvList, return_counts=True)
            plt.bar(keys, counts)
            plt.xticks(rotation='vertical')
            plt.show()

            plt.plot(smList)
            plt.show()
        bestWvlt = max(set(wvList), key=wvList.count)
        print("Best performing wavelet: {}".format(bestWvlt))
        print("Least similarity found: {} in spectra #{}".format(min(smList), smList.index(min(smList))))

    if plot_fig:
        nS = 1520 # np.random.randint(spectra_array.shape[0])
        signal = copy.deepcopy(spectra_array[nS, :])
        filtered = wavelet_denoising(signal, wavelet=bestWvlt)  #[0:-1]
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
        fig.suptitle('wavelet smoothing spec: #{}'.format(nS), fontsize=12, y=1, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.tight_layout(pad=0.2)
        plt.show()

    if smoothed_array: # already picked and bestWvlt is None:
        smoothed_array_ = np.zeros_like(spectra_array)
        for nS in range(spectra_array.shape[0]):
            signal = copy.deepcopy(spectra_array[nS, :])
            filtered = wavelet_denoising(signal, wavelet=bestWvlt)  #[0:-1] #wvList[nS]) #TODO: odd/even size
            print(">> ", spectra_array.shape)
            print(">> ", signal.shape, filtered.shape)
            break
            assert (signal.shape == filtered.shape)
            filtered = zeroBaseSpec(filtered)
            smoothed_array_[nS, :] = filtered
        return bestWvlt, smoothed_array_
    else:
        return bestWvlt, smList

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

def matchSpecLabel(seg1, seg2, arr1, arr2, plot_fig=True): #Done: implement for multi seg
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

def matchSpecLabel2(plot_fig, *segs, **kwarr): # exprun
    """
    seg1 & seg2: segmentation file path(.npy)
    Comparison of segmentation between two region arrays' (3D) mean
    exprun: string command, name the experiment run
    e.g.: matchSpecLabel2(True, seg1_path, seg2_path, seg3_path, seg4_path, arr1=spec_array1,
                                                                        arr2=spec_array2,
                                                                      arr3=spec_array3,
                                                                    arr4=spec_array4)
    """
    specDict = {}
    segList = []
    title = []
    for i, (s, (key, value)) in enumerate(zip(segs, kwarr.items())):
        label = np.load(s)
        # print(np.unique(label))
        segList.append(label)
        title.append(key)
        for l in range(1, len(np.unique(label))): # 0 is background
            label_ = copy.deepcopy(label)
            label_[label != l] = 0
            spec = np.mean(value[np.where(label_)], axis=0)
            spec = {"{}_{}".format(i+1, l): spec}
            specDict.update(spec)
    print(specDict.keys())
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
    ax.set_title("{}".format(kwarr['exprun']))
    fig.show()

    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = mtl.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
        return texts

    # fig, ax = plt.subplots()
    # im, cbar = heatmap(corr, specDict.keys(), specDict.keys(), ax=ax,
    #                    cmap="YlGn", cbarlabel="harvest [t/year]")
    # ax.xaxis.tick_top()
    # texts = annotate_heatmap(im, valfmt="{x:.2f}")
    #
    # fig.tight_layout()
    # # plt.colorbar()
    # plt.show()

    # def getSub(i):
    c_ = i + 1  # how many columns in subplots
    if (i + 1) % c_ == 0:
        r_ = (i + 1) // c_
    else:
        r_ = (i + 1) // c_ + 1
    # return r_, c_
    if plot_fig:
        fig, axs = plt.subplots(r_, c_, figsize=(16, 16), dpi=300, sharex=False)
        for ar, tl, ax in zip(segList, title, axs.ravel()):
            im = ax.imshow(ar) #, cmap='twilight') #cm)
            ax.set_title(tl, fontsize=10)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, ax=ax)
        plt.suptitle("{}".format(kwarr['exprun']))
        plt.show()
    return specDict

# def rawVSprocessed(rawMSpath, proMSpath):
def rawVSprocessed(mzraw, abraw, mzpro, abpro, labels=None, n_spec=None, exprun=None):
    if exprun is None:
        exprun = 'demo'
    if labels is None:
        labels = ['raw', 'processed']

    fig, ax = plt.subplots(2, 1, dpi=100)

    ax[0].hist(mzraw, color=(0.9, 0, 0), linewidth=1.5, label=labels[0], bins=200)  # , alpha=0.9)
    ax[0].set_xlabel("m/z", fontsize=12)
    ax[0].set_ylabel("counts", fontsize=12, color=(0.9, 0, 0))
    ax[0].legend(loc='upper center')
    # ax[0].grid()

    ax0 = ax[0].twinx()
    ax0.hist(mzpro, color=(0.0, 0, 0.9), linewidth=1.5, label=labels[1], bins=200, alpha=0.5)
    ax0.set_xlabel("m/z", fontsize=12)
    ax0.set_ylabel("counts", fontsize=12, color=(0.0, 0, 0.9)) #, alpha=0.5)
    ax0.legend(loc='upper right')

    # ax[1].plot(mzraw, abraw, color=(0.9, 0, 0), linewidth=1.5, label=labels[0])  # , alpha=0.9)
    # ax[1].plot(mzraw[rawz_], abraw[rawz_], marker='r^')
    ax[1].vlines(mzraw, ymin=[0], ymax=abraw, color=(0.9, 0, 0), linewidth=1.5, label=labels[0]) #, alpha=0.5)
    ax[1].set_xlabel("m/z", fontsize=12)
    ax[1].set_ylabel("intensity", fontsize=12, color=(0.9, 0, 0))
    ax[1].legend(loc='upper center')
    # ax[1].grid()

    ax1 = ax[1].twinx()
    # ax1.plot(mzpro, abpro, color=(0, 0, 0.9), linewidth=1.5, label=labels[1], alpha=0.5)
    # ax1.plot(mzpro[proz_], abpro[proz_], marker='bo')
    ax1.vlines(mzpro, ymin=[0], ymax=abpro, color=(0, 0, 0.9), linewidth=1.5, label=labels[1], alpha=0.3)
    ax1.set_xlabel("m/z", fontsize=12)
    ax1.set_ylabel("intensity", fontsize=12, color=(0, 0, 0.9)) #, alpha=0.5)
    ax1.legend(loc='upper right')
    # ax1[1].grid()
    ax[0].set_title("Binning - histogram")
    ax[1].set_title("Spectrum")
    if n_spec is not None:
        fig.suptitle('A spectrum representation #{} {}'.format(n_spec, exprun), fontsize=14, y=0.99)
    else:
        fig.suptitle('A spectrum representation {}'.format(exprun), fontsize=14, y=0.99)
    # fig.subplots_adjust(top=0.85)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout(pad=0.2)
    plt.show()




