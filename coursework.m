clear all
close all


%colourMatrix('SimulatedImages/org_1.png', false, false)
colourMatrix('SimulatedImages/noise_3.png', false, true)
%colourMatrix('SimulatedImages/proj2_1.png', false, false)
%colourMatrix('SimulatedImages/rot_1.png', false, false)

function [results] = colourMatrix(filename, displaySteps, noisy)

    %----------------Read in image---------------------
    img = imread(filename);
    figure('Name', ' Original Image'), imshow(img)
    fixedPoints = [26.5 26.5; 26.5 445.5; 445.5 26.5; 445.5 445.5];

    %========================== TRANSFORMATION ==========================

    %----------------Convert to B+W--------------------
    BW = ~im2bw(img, 0.5); %invert
    if displaySteps
        figure('Name', 'inv B&W'), imshow(BW) 
    end
 
    %------------------Errode----------------------------
    %se = strel('disk',2); %create strel obj to erode with
    %erodeBW = imerode(BW, se); %erode to clear circles
    %imshow(erodeBW);

    %----------------Fill holes-------------------------
    filled = imfill(BW,'holes');
    if displaySteps
        figure('Name', 'Filled'), imshow(filled);
    end

    %---------------Identify Cirlces--------------------
    CC=bwconncomp(BW); %Create data struct, used to be applied to filled
    s = regionprops(filled,'Eccentricity', 'Centroid', 'Area'); %ID circles
    maximum = max([s.Area]);
    %Remove max area as this is the square
    for x = 1:size(s,1)
        if s(x).Area == maximum
            s(x) = [];
            break
        end
    end

    %--------------Retrieve Centroids------------------
    centroids = [s.Centroid];
    centres = vec2mat(centroids,2); %can reduce to 1 line
    movingPoints = centres;

    %------------Create Transformation Matrix---------
    mytform = fitgeotrans(movingPoints,fixedPoints, 'projective');
    mytform.T;

    %-----------Apply to original--------------------
    %figure(1), imshow(img)
    imgTrans=imwarp(img,mytform);
    %figure(2)
    if displaySteps
        figure('Name', 'Transformed'), imshow(imgTrans)
    end

    %-----------------Crop---------------------
    % ctrs = findCrop(imgTrans);
    % imgCropped = imcrop(imgTrans,[ctrs(1) ctrs(2) 420 420]);
    % figure('Name', 'Cropped'), imshow(imgCropped)



    %========================== LOCATE TILES ==========================

    %----------------Convert to B+W--------------------
    if noisy %Adjust for noisy images
        threshold = 0.4;
    else
        threshold = 0.1;
    end
    BW = im2bw(imgTrans, threshold); %Convert to B&W around threshold
    if displaySteps
        figure('Name', 'Transformed B&W'), imshow(BW)
    end

    %------------------Dilate----------------------------
    % se = strel('disk',1); %create strel obj to dilate with
    % dilateBW = imdilate(BW, se); %erode to clear circles
     %figure('Name', 'Dilated'), imshow(dilateBW);

    %------------------Errode----------------------------
    se = strel('square',5); %create strel obj to erode with (set to 10 for noisy imgs)
    erodeBW = imerode(BW, se); %erode to clear circles
    if displaySteps
        figure('Name', 'Erroded'), imshow(erodeBW);
    end

    %---------------Identify Tile Centroids---------------------
    t = regionprops(erodeBW, 'Centroid', 'Area'); %ID circles
    med = median([t.Area]); %Average not suitable
    margin = med * 0.1; 
    %Remove all objects with areas 10% from median
    x = 1;
    while size(t,1) > 16
        if t(x).Area < med-margin || t(x).Area > med+margin
            t(x) = [];
        else
            x = x + 1; %only increment if entry not removed
        end
    end


    %======================== COLOUR IDENTIFICATION =========================

    %------------------------Lab values of colours---------------------------
    white = [128,128]; %128,128
    blue = [145,95];
    yellow = [116,195]; 
    green = [65,180]; 
    red = [170,150]; 
    colours = [white; blue; yellow; green; red]; %colour index = position

    %----------------------Median filter rgb channels------------------------
    C = makecform('srgb2lab');
    medianFilt = imgTrans;
    medianFilt(:,:,1) = medfilt2(medianFilt(:,:,1), [9 9]);
    medianFilt(:,:,2) = medfilt2(medianFilt(:,:,2), [9 9]);
    medianFilt(:,:,3) = medfilt2(medianFilt(:,:,3), [9 9]);

    %---------------------------Lab colour space---------------------------
    imgLab = applycform(medianFilt, C); % convert to lab
    if displaySteps
        figure('Name', 'Median'), imshow(medianFilt);
        figure('Name', 'Median+Lab'), imshow(imgLab);
    end

    %--------------Seperate img in to seperate component layers-----------
    L = imgLab(:,:,1); %L = lightness, comment on live imgs?
    a = imgLab(:,:,2); %A = Red/Green val
    b = imgLab(:,:,3); %B = Blue/Yello val

    %-----------Create Datastructures to store tile averages-------------
    aGrid = [];
    bGrid = [];
    lGrid = [];
    colGrid = [];

    %-----------------------Iterate over each tile----------------------
    for i = 1:16
        tileDim = sqrt(t(i).Area) * 0.7; %Area to be averaged is 70% of the tile
        centre = round(t(i).Centroid);

        %Identify 3 corners of tile area to be averaged, relative to centre
        A = centre + [round((-1/2) * tileDim), round((-1/2) * tileDim)]; %Top left corner
        B = centre + [round((1/2) * tileDim), round((-1/2) * tileDim)]; %Bottom left
        C = centre + [round((-1/2) * tileDim), round((1/2) * tileDim)]; %Top right

        %Calc avg L, A and B values within each tile
        avgL = mean(mean(L(A(1):B(1), A(2):C(2))));
        avgA = mean(mean(a(A(1):B(1), A(2):C(2))));
        avgB = mean(mean(b(A(1):B(1), A(2):C(2))));
        avs = [avgA, avgB];

        %Collect Values
        lGrid = [lGrid, avgL];
        aGrid = [aGrid, avgA];
        bGrid = [bGrid, avgB];

        %Minimum Euclidean distance to org colours
        [~,colcode] = min(sqrt(sum((colours-avs).^2,2)));
        colGrid = [colGrid, colcode];
    end

    %-----------------------Reformat to 4x4 Grid-----------------------------
    %lGrid = reshape(lGrid,[4,4]);
    aGrid = reshape(aGrid,[4,4]);
    bGrid = reshape(bGrid,[4,4]);
    colGrid = reshape(colGrid, [4,4]);
    %disp(colGrid');

    %-----------------Translate Colour Indexes to String--------------------
    results = strings(4,4);
    for i = 1:4
        for j = 1:4
            switch colGrid(i,j)
                case 1
                    results(i,j) = 'White';
                case 2
                    results(i,j) = 'Blue';
                case 3
                    results(i,j) = 'Yellow';
                case 4
                    results(i,j) = 'Green';
                case 5
                    results(i,j) = 'Red';
            end   
        end
    end
    %Transpose to match original
    results = results';
    %Output Results  
    disp(results);
end





%Used to locate top left centroid to use as crop reference
function [centroids] = findCrop(img)
    %figure(1), imshow(img);
    maximum = 0;
    %----------------Convert to B+W-----------
    BW = ~im2bw(img, 0.5); %invert
    %figure(2), imshow(BW);
    %----------------Fill holes---------------
    filled = imfill(BW,'holes'); 
    %figure(3), imshow(filled);
    %---------------Identify Cirlces--------------------
    CC=bwconncomp(BW); %Create data struct
    info = regionprops(BW,'Eccentricity', 'Centroid', 'Area'); %ID circles
    %disp(s.Area);
    areas = info.Area;
    disp(areas);
    maximum = max(areas);
    %Remove max area as this is the square
    for x = 1:size(info,1)
        if info(x).Area == maximum
            info(x) = [];
            break
        end
    end
    %--------------Retrieve Centroids------------------
    centroids = vec2mat([info.Centroid],2);
    %centroids = centroids(1,:);
end


