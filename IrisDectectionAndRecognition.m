%FIT3081 Assignment 2 
%Iris Detection and Recognition

% initialising and loading file path directory
folder = 'iris_dataset';
imagefiles = dir(folder);
nFiles = length(imagefiles); % number of image +2 

template_cell = cell(nFiles-2,1);

% loop thru all images in directory
% does image processing up until convolution with log gabor filter for all images 
% to produce a template and store in a cell array

% first two index are irrevelant so read from index 3
for i=3:nFiles
    currentFilename = imagefiles(i).name;
    currentimage = imread(fullfile(folder,currentFilename));
    
    % image preprocessing
    I = rgb2gray(currentimage);
    I = imresize(I,.75);
    I = medfilt2(I);
    
    % produce edge map and get circles using Hough Transform
    cannyMap = edge(I, 'canny');
    [centers_pupil, radii_pupil] = imfindcircles(cannyMap, [20 55], 'Sensitivity',0.90);
    [centers_iris, radii_iris] = imfindcircles(cannyMap, [60 150], 'Sensitivity',0.95);
    
    % setting values for image 
    xIris = centers_iris(1);
    yIris = centers_iris(2);
    rIris = radii_iris;
    xPupil = centers_pupil(1);
    yPupil = centers_pupil(2);
    rPupil = radii_pupil;
    
    % normalisation then histogram equliser
    normalisedImage = normalisation(I, xIris, yIris, rIris, xPupil, yPupil, rPupil);
    equalisedImage = histeq(normalisedImage);
    
    % finally log gabor filter to produce template
    template_cell{i-2} = logGaborFilter(equalisedImage);
    
end

% call for pairwise hamming distance between all images and stores it in
% matrix call hamDistMatrix
lenTemplate = nFiles - 2;
hamDistMatrix = zeros(lenTemplate-1, lenTemplate);

for i=1:lenTemplate-1
    start = i+1;
    for j=start:lenTemplate
        hamDistMatrix(i,j) = hammingDistance(template_cell{i,1},template_cell{j,1});
    end
end

%extract non zero elements from hamDistMatrix and plot histogram
nonZero = hamDistMatrix~=0;
distanceArray = hamDistMatrix(nonZero);
%histogram(distanceArray);

% calculate mean and standard deviation to decide on threshold
% ham dist below threshold = same iris
% ham dist above threshold = diff iris

hamDistMean = mean2(distanceArray)
hamDistStD = std(distanceArray)
threshold = hamDistMean-(2*hamDistStD)
threshold = round(threshold,1)

% calculate True Positive, False Negative, False Positive and True Negative of output
sw = 3;
TP = 0; FN = 0; FP = 0; TN = 0;

for i=1:lenTemplate
    for j=1:lenTemplate-1
        if hamDistMatrix(j,i) == 0
            break
        else
            % if images belong to the same iris
            if j >= (sw-2) && j <= sw
                % estimate positive when true value is positive
                if hamDistMatrix(j,i) <= threshold
                    TP = TP + 1;
                % estimate negative when true value is positive
                else
                    FN = FN + 1;
                end
            % if images does not belong to the same iris  
            else  
                % estimate positive when true value is negative
                if hamDistMatrix(j,i) <= threshold
                    FP = FP + 1;
                % estimate negative when true value is negative
                else
                    TN = TN + 1;
                end
            end
        end
    end
    if rem(i,3) == 0
        sw = sw + 3;
    end
end

display(TP); display(FN); display(FP); display(TN); 

% Recall formula
% calculate accuracy for detecting if two images are from SAME iris
same_accuracy = (TP/(TP+FN))*100;
display(same_accuracy);

% Specificity formula
% calculate accuracy for detecting if two images are from DIFFERENT iris
diff_accuracy = (TN/(FP+TN))*100;
display(diff_accuracy);

function normalisedImage = normalisation(image, xIris, yIris, rIris, xPupil, yPupil, rPupil)
    
    offsetX = yIris - yPupil;
    offsetY = xIris - xPupil;
    
    angleSize = 360; % these values are basically the size of the normalised image. setting them smaller will go faster due to less pixel processing
    radiusSize = 45; 
    
    normalisedImage = zeros(radiusSize, angleSize);
    for t=1:angleSize
        radiusColumn = zeros(radiusSize,1);
        for r=1:radiusSize
            currAngle = (t-1)*2*pi/angleSize + pi;
            
            % calculates distance between edge of iris and edge of pupil for every angle and radius
            % based on daugman's formula
            a = offsetX^2 + offsetY^2;
            b = cos(pi-atan(offsetY/offsetX)-angleSize);
            dist = sqrt(a)*b + sqrt(rIris^2 - a + a*b^2);

            % get the cordinate values for interpolation
            ro = (dist - rPupil)*r/radiusSize + rPupil;
            x = xIris - (ro)*sin(currAngle);
            y = yIris + (ro)*cos(currAngle);

            % interpolation to determine intensity values of normalised image
            radiusColumn(r,1) = interp2(double(image), x, y, "linear");
    
        end
        normalisedImage(:,t) = radiusColumn;
    end
    normalisedImage = uint8(normalisedImage);
end


function template = logGaborFilter(normalised_image)

    [rows,cols] = size(normalised_image);   
    
    % initialisation of filter array and result array
    logGabor = zeros(1,cols);
    result = zeros(rows,cols);
    
    % initialise values to calculate log gabor filter
    radius =  (0:fix(cols/2))/fix(cols/2)/2;  
    radius(1) = 1;
    fo = 1/18;    % Centre frequency of filter = 1/wavelength where wavelength initialised to 18
    sigmaOfFrequency = 0.5;
    
    % calculate the filter
    logGabor(1:cols/2+1) = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOfFrequency)^2));  
    logGabor(1) = 0;  
    
    % for each row of the input image, perform convolution
    for r = 1:rows
        signal = normalised_image(r,1:cols);
        imagefft = fft(signal); %give frequency information of the signal 
        result(r,:) = ifft(imagefft .* logGabor); %inverse back
    end

    template = zeros(size(normalised_image,1), cols*2);
    h = 1:size(normalised_image,1);
    
    % splitting real and imaginary for complex number
    realN = real(result) > 0;
    imaginaryN = imag(result) > 0;
    
    % construct the biometric template using phase quantisation of real and
    % imaginary component
    for i=0:(cols-1)
        j = double(2*(i));
        template(h,j+1) = realN(h, i+1);
        template(h,j+2) = imaginaryN(h,i+1);
    end

end

function dist = hammingDistance(gaborTemplate1, gaborTemplate2)
    % preparing both templates to be compared
    template1 = logical(gaborTemplate1);
    template2 = logical(gaborTemplate2);
    
    % calculate total number of bits in template
    totalBits = size(template1,1)*size(template1,2);
    
    % perform XOR of both template
    templateXOR = xor(template1,template2);
    
    % obtain number of differet bits in both template
    difference = sum(templateXOR==1);
    differenceBits = sum(difference);
    
    % hamming distance = difference / total
    dist = differenceBits/totalBits;
end