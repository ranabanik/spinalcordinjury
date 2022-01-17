require(Cardinal)
require(dplyr)

msdata <- readImzML("210427-Chen_poslip",mass.range=c(510,2000),resolution=3, folder="C:/Users/pattenh1/Downloads/",)

# generate mean spectrum from all spectra at 3 ppm mass resolution
mean_spec <- msdata %>%
  summarizeFeatures(FUN="mean")

# peak pick mean spectrum to find peaks
mse_peakref <- mean_spec %>%
  peakPick() %>%
  peakFilter() %>%
  process()

# bin raw spectra to peak picked mass spectra
# tolerance to search for peak set at 2x mass res.
mspeaks <- msdata %>%
  peakBin(ref=mse_peakref@featureData@mz,tolerance=6) %>%
  process()

image(mspeaks, mz=760.58)
image(mspeaks, mz=734.55)
image(mspeaks, mz=703.58)

# segment with k=4
spatial_seg <- spatialShrunkenCentroids(mspeaks, r=1, k=4, s=1)
# image segmentation
image(spatial_seg)

# plot segmentation top values per clus
plot(spatial_seg, values="statistic")
topFeatures(spatial_seg)
# view image of a top feature
image(mspeaks, mz=788.61)

writeImzML(mspeaks, "210427-Chen_poslip-peakpicked", "D:/temp")