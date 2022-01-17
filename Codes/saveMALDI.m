% close all;
clear all;
warning off;
clc;
%%{
% pathN =  '/media/banikr2/banikr/MALDI_protein/'; 
pathN = '/media/banikr2/DATA/MALDI/demo_banikr_/';
% load([pathN 'demo_chen_protein.mat']); 
load([pathN 'demo_chen_pos_lip.mat']);
imSize = [length(MSi.XData) length(MSi.YData)];
%%
k = 1;
for c = 1:prod(imSize)
    if ~isempty(MSi.MSo.scan(c).mz)
        kps(k) = c; 
        k = k + 1;
    end
end
%%
A = zeros(imSize);
A(kps) = 1;
p = [1 imSize(1) 1 imSize(2)]; % content range in image
k = 1; % this creates a binary mask. 
while(sum(A(k, :)) == 0)
k = k + 1;
end
p(1) = k;
k = imSize(1);
while(sum(A(k, :)) == 0)
    k = k - 1;
end
p(2) = k;
k = 1;
while(sum(A(:, k)) == 0)
    k = k + 1;
end
p(3) = k;
k = imSize(2);
while(sum(A(:, k)) == 0)
    k = k - 1;
end
p(4) = k;
%%
B = A(p(1):p(2), p(3):p(4));
%%
keeps = find(A > 0);
imSize = size(A);
for c = 1:length(keeps)
    cc = kps(c);
    if isempty(MSi.MSo.scan(cc).mz)
        mz{c} = 0;
        mz_mean(c) = 0;
        abndc{c} = 0;
        abndc_mean(c) = 0;
        numpeaks(c) = 0;
        peakmz(c) = 0;
        peakAbndnc{c} = 0;
        peakAbndnc_mean(c) = 0;
        ionCrntT(c) = 0;
        ionCrntLocal{c} = 0;
        ionCrntLocal_mean(c) = 0;
    else
        mz{c} = MSi.MSo.scan(cc).mz;
        mz_mean(c) = mean(MSi.MSo.scan(cc).mz);
        abndc{c} = MSi.MSo.scan(cc).abundance;
        abndc_mean(c) = mean(MSi.MSo.scan(cc).abundance);
        numpeaks(c) = MSi.MSo.scan(cc).numpeaks;
        peakmz(c) = MSi.MSo.scan(cc).peakmz;
%         peakAbndnc{c} = MSi.MSo.scan(cc).peakabundance;
        peakAbndnc(c) = MSi.MSo.scan(cc).peakabundance;
        peakAbndnc_mean(c) = mean(MSi.MSo.scan(cc).peakabundance);
        ionCrntT(c) = MSi.MSo.scan(cc).totalioncurrent;
        ionCrntLocal{c} = MSi.MSo.scan(cc).localioncurrent;
        ionCrntLocal_mean(c) = mean(MSi.MSo.scan(cc).localioncurrent);
    end
end
A(keeps) = peakAbndnc; %ionCrntT;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;
% save spineMALDI.mat keeps imSize mz mz_mean abndc abndc_mean numpeaks peakmz peakAbndnc peakAbndnc_mean ...
%         ionCrntT ionCrntLocal ionCrntLocal_mean;
%}
%%
load spineMALDI.mat
reslN = 100;
interp_method = 'linear';
v1 = ones(round(50*imSize(1)/reslN), round(50*imSize(2)/reslN));
v2 = ones(imSize(1), imSize(2));
[x y X Y] = interpAnatFnc(v1, v2);

A = zeros(imSize);
A(keeps) = abndc_mean;
B = interp2(X, Y, A, x, y, interp_method);
figure;
imagesc(permute(B, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = peakAbndnc_mean;
B = interp2(X, Y, A, x, y, interp_method);
figure;
imagesc(permute(B, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = ionCrntT;
B = interp2(X, Y, A, x, y, interp_method);
figure;
imagesc(permute(B, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = ionCrntLocal_mean;
B = interp2(X, Y, A, x, y, interp_method);
figure;
imagesc(permute(B, [2 1]));
axis off;
colormap(jet);
colorbar;

%{
A = zeros(imSize);
A(keeps) = abndc_mean;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = peakAbndnc_mean;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = ionCrntT;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = ionCrntLocal_mean;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = mz_mean;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = numpeaks;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;

A(keeps) = peakmz;
figure;
imagesc(permute(A, [2 1]));
axis off;
colormap(jet);
colorbar;
%}



































