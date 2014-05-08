%   Distribution code Version 1.0 -- Oct 12, 2013 by Cewu Lu 
%
%   The Code is to demo Sparse Combination in our Avenue Dataset, based on the method described in the following paper 
%   [1] "Abnormal Event Detection at 150 FPS in Matlab" , Cewu Lu, Jianping Shi, Jiaya Jia, 
%   International Conference on Computer Vision, (ICCV), 2013
%   
%   The code and the algorithm are for non-commercial use only.

%% abnormal event detection in testing
addpath('functions')
addpath('data')
load('data/sparse_combinations/Tw.mat','Tw');
load('data/sparse_combinations/R.mat','R');
params.H = 90;       % loaded video height size
params.W = 160;       % loaded video width size
params.patchWin = 5; % 3D patch spatial size 
params.tprLen = 5;    % 3D patch temporal length
params.BKH = 18;      % region number in height
params.BKW = 32;      % region number in width
params.srs = 3;       % spatial sampling rate in trainning video volume
params.trs = 2;       % temporal sampling rate in trainning video volume 
params.PCAdim = 100;  % PCA Compression dimension
params.MT_thr = 2;    % 3D patch selecting threshold 

H = params.H;
W = params.W; 
patchWin = params.patchWin;
tprLen = params.tprLen; 
BKH = params.BKH;
BKW = params.BKW;
PCAdim = params.PCAdim;
ThrTest = 0.22;
ThrMotionVol = 2; 
 


%volFrame = 10; % the number of video we test
%volFrame = 20;
%volFrame = 21;

load('data/CV_Abnormality_8.mat'); 
%imgVol = im2double(vol);

for ii = 1 : size(Video_Output, 4)
    Video_Output(:, :, :, ii) = Video_Output(:, :, :, ii)/255;
    imgVol(:, :, ii) = rgb2gray(Video_Output(:, :, :, ii));
end

t1 = tic;
volBlur = imgVol; 
blurKer = fspecial('gaussian', [3,3], 1);
mask = conv2(ones(H,W), blurKer,'same');
for pp = 1 : size(imgVol,3)
     volBlur(:,:,pp) =  conv2(volBlur(:,:,pp), blurKer, 'same')./mask;
end
feaVol = abs(volBlur(:,:,1:(end-1)) - volBlur(:,:,2:end));
[feaPCA, LocV3] = test_features(feaVol, Tw, ThrMotionVol, params); 
Err = recError(feaPCA, R, ThrTest);

AbEvent = zeros(BKH, BKW, size(imgVol,3));
for ii = 1 : length(Err)
    AbEvent(LocV3(1,ii),LocV3(2,ii),LocV3(3,ii)) =  Err(ii);
end
AbEvent3 = smooth3( AbEvent, 'box', 5);
% AbEvent3 already can indicate abnormalities in each region, so testing stops here.
t2 = toc(t1); 
fprintf('We can achieve %d FPS in the current testing video\n', round(size(imgVol,3)/t2));


%% video demo
optThr = 0.15;
AbEventShow3 = imgVol; 
for frameID = 1 : size(imgVol,3)
    AbEventShow3(:,:,frameID) = double(imresize(AbEvent3(:,:,frameID) ,[H, W], 'nearest') > optThr) ;
end

grid = zeros(H, W);
grid(:, [1, patchWin: patchWin: W]) = 1;
grid([1, patchWin: patchWin: H], :) = 1;

for frameID = 1 : size(Video_Output,4)  
    curFrame = Video_Output(:, :, :, frameID);
    curFrame(:, :, 2) = min(curFrame(:, :, 2) + 0.5 * AbEventShow3(:,:,frameID), 1);
    %curFrame(:, :, 3) = min(curFrame(:, :, 3) + 0.95 * grid, 1);
    curFrame = imresize(curFrame, 3);
    imshow(curFrame);
    pause(1/100);
    %getframe;
end