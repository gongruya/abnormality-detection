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

set_params;
params.MT_thr = 2;

H = params.H;
W = params.W; 
patchWin = params.patchWin;
tprLen = params.tprLen; 
BKH = params.BKH;
BKW = params.BKW;
PCAdim = params.PCAdim;
ThrTest = 0.24;


%volFrame = 10; % the number of video we test
%volFrame = 20;
%volFrame = 21;

load('data/CV_Bicycle_on_the_Lane.mat'); 
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
[feaPCA, LocV3] = test_features(feaVol, Tw, params); 

AbEvent = zeros(BKH, BKW, size(imgVol,3));

mask = conv2(ones(BKH, BKW), blurKer, 'same');

for ii = 1 : size(feaPCA, 2)
    ab = 1;
    for jj = 1 : length(R)
        if norm(R(jj).val * feaPCA(:, ii))^2 <= ThrTest
            ab = 0;
            break;
        end
    end
    AbEvent(LocV3(1,ii),LocV3(2,ii),LocV3(3,ii)) = ab;
end
AbEvent = smooth3(AbEvent, 'box', 5) > 0.1;
t2 = toc(t1); 
fprintf('We can achieve %d FPS in the current testing video\n', round(size(imgVol,3)/t2));


%% video demo
AbEventShow3 = imresize(AbEvent, [H, W], 'nearest');

grid = zeros(H, W);
grid(:, [1, patchWin: patchWin: W]) = 1;
grid([1, patchWin: patchWin: H], :) = 1;
grid_rgb = zeros(H, W, 3);
grid_rgb(:, :, 1) = 0.84 * grid;                %r
grid_rgb(:, :, 2) = 0.94 * grid;                %g
grid_rgb(:, :, 3) = 0.14 * grid;                %b
grid_rgb = imresize(grid_rgb, 3, 'nearest');

for frameID = 1 : size(Video_Output,4) - 2
    curFrame = Video_Output(:, :, :, frameID);
    curFrame(:, :, 2) = min(curFrame(:, :, 2) + 0.8 * AbEventShow3(:,:,frameID), 1);
    curFrame = imresize(curFrame, 3);
    curFrame = min(curFrame + 0.3 * grid_rgb, 1);
    imshow(curFrame);
    text(100, 20, [num2str(frameID) '/' num2str(size(Video_Output,4))], ...
        'FontSize', 20, 'FontName', 'Courier New', 'FontWeight', 'bold', 'BackgroundColor', [.7 .9 .7], 'HorizontalAlignment', 'center');
    pause(1/200);
    %getframe;
end