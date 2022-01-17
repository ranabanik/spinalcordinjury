%% Load files
clear all
close all

[file1,path1]=uigetfile('/home/banikr/*.fdf','Open .fdf file','MultiSelect','on');
    if iscell(file1)==0 % If the output variable of the file names opened is a cell, then it needs to be rearranged to a column array
    disp(file1)
    else 
    disp(file1')
    end
    
cd(path1)   
    
if iscell(file1)==0  % If only 1 .mat file is loaded
    dataname=strcat(path1,file1); % Full path name
    filename=file1;   
elseif iscell(file1)==1 % If muliple .mat files are loaded
    for i=1:length(file1) % # of file names loaded
        dataname{i,:}=strcat(path1,file1{i}); % Full path name
        filename{i,:}=file1{i};

    end
end

dataname=string(dataname); % Converts cell to string array, necessary for later code
filename=string(filename);

%% Load and display
for i = 1:length(dataname)
    path = dataname{i};
    img{i} = fdf_func(path);
    figure;
    imagesc(img{i})
    colormap(gray)
    axis image
    axis off    
end

