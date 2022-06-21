import os
import time
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import copy
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from scipy.io import loadmat, savemat
from scipy import ndimage, signal, interpolate, stats
from scipy.spatial.distance import cosine
import scipy.cluster.hierarchy as sch
import pywt
import mglearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as SS
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP
import hdbscan
import matplotlib as mtl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns
from pyimzml.ImzMLParser import ImzMLParser, _bisect_spectrum
from ms_peak_picker import pick_peaks
# from imzml import IMZMLExtract, normalize_spectrum
from matchms import Spectrum, calculate_scores
from matchms.similarity import CosineGreedy, CosineHungarian, ModifiedCosine
from tqdm import tqdm, tqdm_notebook
import joblib
from collections import defaultdict

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

    def resample_region(self, regID, tol=0.02, savedata=True):
        """
        resamples spectra with equal bin width. Considers m/z range for all sections/regions
        """
        regname = os.path.join(os.path.dirname(self.mspath), 'reg_{}_tol_{}.h5'.format(regID, tol))
        # print(os.path.isfile(regname))
        if os.path.isfile(regname):
            print("Previous upsampling found. Fetching...")
            f = h5py.File(regname, 'r')
            array2D = f['2D']
            massrange = f['mzrange']
            regionshape = f['regionshape']
            lCoorIdx = f['localCoor']
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
            for idx, coord in enumerate(tqdm(regionPixels, desc='binning')):
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
                with h5py.File(regname, 'w') as pfile:
                    pfile['2D'] = array2D
                    pfile['mzrange'] = massrange
                    pfile['regionshape'] = regionshape
                    pfile['localCoor'] = lCoorIdx
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

    def peak_pick(self, spectra, refmz, meanSpec=None):
        """
        spectra: 2D array of spectra -> nSpec x nMZ
        meanSpec: mean abundance/intensity of all regions
        by peak_picking ...
        """
        picking_method = "quadratic"
        snr = 3    # standard: lower value increases number of peaks
        intensity_threshold = 5    #5 # depends on instrument/ 0 -> more permissive
        fwhm_expansion = 1.4    # shouldn't be more than 2; 1.2 - 1.4 is optimum
        if meanSpec is None: #spectra.ndim == 2:
            meanSpec = np.mean(spectra, axis=0)
        # else:   # if mean of all regions given...
        #     meanSpec = spectra
        peak_list = pick_peaks(refmz,
                               meanSpec,
                               fit_type=picking_method,
                               signal_to_noise_threshold=snr,
                               intensity_threshold=intensity_threshold,
                               integrate=False)
        peak_mzs = [np.round(p.mz, 5) for p in peak_list]   # only for visualisation
        peak_ranges = [
            (
                p.mz - (p.full_width_at_half_max * fwhm_expansion),
                p.mz + (p.full_width_at_half_max * fwhm_expansion),
            )
            for p in peak_list]
        # print("peak ranges", peak_ranges)
        peak_indices = [
            (self._find_nearest(refmz, p[0]), self._find_nearest(refmz, p[1])) for p in peak_ranges
        ]
        spectra_ = []
        for spectrum in spectra:
            peaks = []
            for p in peak_indices:
                peaks.append(np.sum(spectrum[p[0]: p[1]]))
            spectra_.append(peaks)
        spectra = np.array(spectra_, dtype=np.float32)
        return spectra, peak_mzs

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

    def get_ion_images(self, regID, array2D=None, mzrange=None, top=True):
        """
        For visualization, to see if sufficient ion images could be generated...
        peak: plot peak images, default: True
        """
        if any(v is None for v in [array2D, mzrange]):
            array2D, mzrange, regionshape,lCoorIdx = self.resample_region(regID, tol=0.02)
        spectra_peak_picked, peakmzs = self.peak_pick(array2D, refmz=mzrange)
        # array2D, longestmz, regionshape, lCoorIdx = self.get_region_data(regID)
        peak3D = np.zeros([regionshape[0], regionshape[1], len(peakmzs)])
        for idx, coord in enumerate(lCoorIdx):
            peak3D[coord[0], coord[1], :] = spectra_peak_picked[idx, :]

        colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
        n_bin = 100
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='simple_list', colors=colors, N=n_bin))
        imgN = 50   # how many images to take and plot
        if top: # top variance images...
            peakvar = []
            for mz in range(len(peakmzs)):
                peakvar.append(np.std(peak3D[..., mz]))
            topmzInd = sorted(sorted(range(len(peakvar)), reverse=False, key=lambda sub: peakvar[sub])[-topN:])
            # topmzInd = sorted(sorted(range(len(peakvar)), reverse=False, key=lambda sub: peakvar[sub])[-imgN-3000:-3000])
        else:   # random
            topmzInd = np.round(np.linspace(0, len(peakmzs) - 1, imgN)).astype(int)
        Nr = 10
        Nc = 5
        heights = [regionshape[1] for r in range(Nr)]
        widths = [regionshape[0] for r in range(Nc)]
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
                images.append(axs[r, c].imshow(peak3D[..., topmzInd[pv]].T, origin='lower',
                                               cmap='simple_list'))  # 'RdBu_r')) #
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
    return array[idx]

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
            retSpectrum = (retSpectrum / specSum) #* len(retSpectrum)
        return retSpectrum

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
    # _, reg_smooth_, _ = bestWvltForRegion(regSpec, bestWvlt='db8', smoothed_array=True, plot_fig=False)
    reg_norm = np.zeros_like(spectra)
    for s in range(nSpecs):
        reg_norm[s, :] = normalize_spectrum(spectra[s, :], normalize='tic')     #reg_smooth_
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

        colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
        n_bin = 100
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='simple_list', colors=colors, N=n_bin))
        Nc = 2
        # MaxPCs = nPCs# + 1
        if MaxPCs % Nc != 0:
            Nr = int((nPCs + 1) / Nc)
        else:
            Nr = int(nPCs/Nc)
         #MaxPCs
        heights = [regionshape[1] for r in range(Nr)]
        widths = [regionshape[0] for r in range(Nc)]
        fig_width = 5.  # inches
        fig_height = fig_width * sum(heights) / sum(widths)
        fig, axs = plt.subplots(Nr, Nc, figsize=(fig_width, fig_height), dpi=600, constrained_layout=True,
                                gridspec_kw={'height_ratios': heights})
        images = []
        pc = 0
        image = copy.deepcopy(pcs)
        from sklearn.preprocessing import minmax_scale
        image = minmax_scale(image.ravel(), feature_range=(10, 255)).reshape(image.shape)
        for r in range(Nr):
            for c in range(Nc):
                # Generate data with a range that varies from one plot to the next.
                arrayPC = np.zeros([regionshape[0], regionshape[1]], dtype=np.float32)
                for idx, coor in enumerate(localCoor):
                    arrayPC[coor[0], coor[1]] = image[idx, pc]
                images.append(axs[r, c].imshow(arrayPC.T, origin='lower',
                                                   cmap='simple_list'))  # 'RdBu_r')) #
                axs[r, c].label_outer()
                axs[r, c].set_axis_off()
                axs[r, c].set_title('PC{}'.format(pc + 1), fontsize=10, pad=0.25)
                fig.subplots_adjust(top=0.95, bottom=0.02, left=0,
                                    right=1, hspace=0.14, wspace=0)
                pc += 1

        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                        or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())
                    im.set_tight_layout('tight')

        for im in images:
            im.callbacks.connect('changed', update)
        fig.suptitle("PC images {}:reg {}".format(filename, regID))
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

    # if __name__ != '__main__':  #TODO: Fix PCA loadings and ion imaging
    #     # loadings = pca.components_.T
    #     loadings = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(x+1) for x in range(nBins)])
    #     print("loadings: ", loadings) #['PC897'])
    #     SL = loadings**2
    #     print("SL \n", SL)
    #     SSL = np.sum(SL, axis=1)
    #     print(SSL)
    #
    #     print(loadings.shape)
    #     SSL = np.sum(loadings ** 2, axis=1)
    #     print(SSL.shape)
    #     mean_imgs = []
    #     total_SSLs = []
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
    # rawMS = IMZMLExtract(rawMSpath)
    # proMS = IMZMLExtract(proMSpath)
    if labels is None:
        labels = ['raw', 'processed']
    # n_spec = 00 #np.random.randint(len(rawMS.parser.intensityLengths))
    # mzraw = rawMS.parser.getspectrum(n_spec)[0]
    # abraw = rawMS.parser.getspectrum(n_spec)[1]
    #
    # mzpro = proMS.parser.getspectrum(n_spec)[0]
    # abpro = proMS.parser.getspectrum(n_spec)[1]
    rawz_ = np.array(abraw).nonzero()
    proz_ = np.array(abpro).nonzero()
    fig, ax = plt.subplots(2, 1, dpi=100)

    ax[0].hist(mzraw, color=(0.9, 0, 0), linewidth=1.5, label=labels[0], bins=200)  # , alpha=0.9)
    ax[0].set_xlabel("m/z", fontsize=12)
    ax[0].set_ylabel("counts", fontsize=12, color=(0.9, 0, 0))
    ax[0].legend(loc='upper center')
    ax[0].grid()

    # ax[1].plot(mzraw, abraw, color=(0.9, 0, 0), linewidth=1.5, label=labels[0])  # , alpha=0.9)
    # ax[1].plot(mzraw[rawz_], abraw[rawz_], marker='r^')
    ax[1].vlines(mzraw, [0], abraw, color=(0.9, 0, 0), linewidth=1.5, label=labels[0])  # , alpha=0.9)
    ax[1].set_xlabel("m/z", fontsize=12)
    ax[1].set_ylabel("intensity", fontsize=12, color=(0.9, 0, 0))
    ax[1].legend(loc='upper center')
    ax[1].grid()

    ax0 = ax[0].twinx()
    ax0.hist(mzpro, color=(0, 0, 0.9), linewidth=1.5, label=labels[1], bins=200, alpha=0.5)
    ax0.set_xlabel("m/z", fontsize=12)
    ax0.set_ylabel("counts", fontsize=12, color=(0, 0, 0.9), alpha=0.5)
    ax0.legend(loc='upper right')
    # ax0[0].grid()

    ax1 = ax[1].twinx()
    # ax1.plot(mzpro, abpro, color=(0, 0, 0.9), linewidth=1.5, label=labels[1], alpha=0.5)
    # ax1.plot(mzpro[proz_], abpro[proz_], marker='bo')
    ax1.vlines(mzpro, [0], abpro, color=(0, 0, 0.9), linewidth=1.5, label=labels[1], alpha=0.5)
    ax1.set_xlabel("m/z", fontsize=12)
    ax1.set_ylabel("intensity", fontsize=12, color=(0, 0, 0.9), alpha=0.5)
    ax1.legend(loc='upper right')
    # ax1[1].grid()
    ax[0].set_title("Binning - histogram")
    ax[1].set_title("Spectra")
    if n_spec is not None:
        fig.suptitle('A spectrum representation #{} {}'.format(n_spec, exprun), fontsize=12, y=1)
    else:
        fig.suptitle('A spectrum representation {}'.format(exprun), fontsize=12, y=1)
    # fig.subplots_adjust(top=0.85)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout(pad=0.2)
    plt.show()




