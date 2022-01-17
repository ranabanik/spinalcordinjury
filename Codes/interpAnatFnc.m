%% Function to interpolate connectivity to anatomic map
function [x y X Y] = interpAnatFnc(Im, map)
    [x, y] = meshgrid(1:size(Im, 2), 1:size(Im, 1));
    mF1 = size(map, 1)/size(Im, 1);
    mF2 = size(map, 2)/size(Im, 2);
    stepSizeX = 0.9/mF1;
    stepSizeY = 0.9/mF2;
    while (length(1:stepSizeX:size(Im, 1)) >= size(map, 1)),
        stepSizeX = stepSizeX + 0.0001;
    end
    while (length(1:stepSizeY:size(Im, 2)) >= size(map, 2)),
        stepSizeY = stepSizeY + 0.0001;
    end
    stepSizeX = stepSizeX - 0.0001;
    stepSizeY = stepSizeY - 0.0001;
    [X, Y] = meshgrid(1:stepSizeY:size(Im, 2), 1:stepSizeX:size(Im, 1));
    
    