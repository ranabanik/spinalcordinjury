require(Cardinal)
require(dplyr)

#msdata <- readImzML("210427-Chen_poslip",mass.range=c(506,2000),resolution=3, folder="/media/banikr2/DATA/MALDI/demo_banikr_",)
#msdata <- readImzML("210622_TC380_Chen_Rat", mass.range=c(345,3000), resolution=3, folder="/media/banikr2/DATA/MALDI/demo_banikr_/210622_TC380_Chen_Rat",)
msdata <- readImzML("210603-Chen_protein_slide_F", mass.range=c(1593,23215), resolution=3, folder="/media/banikr2/DATA/MALDI/210603-Chen_protein_slide_F",)

#/media/banikr2/DATA/MALDI/demo_banikr_/210622_TC380_Chen_Rat/210622_TC380_Chen_Rat.imzML
# set.seed(2)
# plot(msdata@featureData@mz, msdata@imageData[[1]][,9])
# data <- simulateImage(preset=1, npeaks=10, dim=c(3,3))

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

#image(mspeaks, mz=1760.58)
#image(mspeaks, mz=20734.55)
#image(mspeaks, mz=3703.58)

# segment with k=4
spatial_seg <- spatialShrunkenCentroids(mspeaks, r=1, k=4, s=1)
# image segmentation
image(spatial_seg)

# plot segmentation top values per clus
plot(spatial_seg, values="statistic")
topFeatures(spatial_seg)
# view image of a top feature
#image(mspeaks, mz=12788.61)

writeImzML(mspeaks, "210603-Chen_protein_slide_F", "/media/banikr2/DATA/MALDI/fromCardinal/Protein")