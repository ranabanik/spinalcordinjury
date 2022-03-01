addpath('/media/banikr2/DATA/MALDI/demo_banikr_')
dataPath = '/media/banikr2/DATA/MALDI/demo_banikr_';
m = dir(fullfile(dataPath, '*.mat'))
% m = dir(dataPath, '*.mat')
load(m.name)