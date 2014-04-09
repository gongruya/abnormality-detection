%   Distribution code Version 1.0 -- Oct 12, 2013 by Cewu Lu 
%
%   The Code is to demo Sparse Combination in our Avenue Dataset, based on the method described in the following paper 
%   [1] "Abnormal Event Detection at 150 FPS in Matlab" , Cewu Lu, Jianping Shi, Jiaya Jia, 
%   International Conference on Computer Vision, (ICCV), 2013
%   
%   The code and the algorithm are for non-commercial use only.


%% Parameters 
params.H = 120;       % loaded video height size
params.W = 160;       % loaded video width size
params.patchWin = 10; % 3D patch spatial size 
params.tprLen = 5;    % 3D patch temporal length
params.BKH = 12;      % region number in height
params.BKW = 16;      % region number in width
params.srs = 5;       % spatial sampling rate in trainning video volume
params.trs = 2;       % temporal sampling rate in trainning video volume 
params.PCAdim = 100;  % PCA Compression dimension
params.MT_thr = 5;    % 3D patch selecting threshold 


H = params.H;
W = params.W; 
patchWin = params.patchWin;
tprLen = params.tprLen; 
BKH = params.BKH;
BKW = params.BKW;
PCAdim = params.PCAdim;
testFileNum = 21;

addpath('functions')
addpath('data')

%% Training feature generation (about 1 minute)
 tic;
fileName = 'data/training_vol';
numEachVol = 7000; % The maximum sample number in each training video is 7000 
trainVolDirs = name_filtering(fileName); 
Cmatrix = zeros(tprLen*patchWin^2, length(trainVolDirs)*numEachVol);
rand('state', 0);
for ii = 1 : length(trainVolDirs)
    [feaRawTrain, LocV3Train]  = train_features([fileName,'/', trainVolDirs{ii}], params);
    t = randperm(size(feaRawTrain,2));
    curFeaNum = min(size(feaRawTrain,2),numEachVol);
    Cmatrix(:, numEachVol*(ii - 1) + 1 : numEachVol*(ii - 1) + curFeaNum) =  feaRawTrain(:,t(1:curFeaNum));     %put random curFeaNum column into Cmatrix
    disp(['Feature extraction in ', num2str(ii),' th training video is done!'])
end
Cmatrix(:,sum(abs(Cmatrix)) == 0) = [];         %take out the zero valued columns

COEFF = princomp(Cmatrix');                     %compress raws
Tw = COEFF(:,1:PCAdim)';
feaMatPCA = Tw*Cmatrix;  
save('data/sparse_combinations/Tw.mat','Tw');
 toc;

%% Sparse combination learning  (about 4 minutes)
tic;
D = sparse_combination(feaMatPCA, 20, 0.1);
%   D = sparse_combination(X, Dim, Thr) learns sparse combination 
%
%   input: 
%   @X: feature matrix m x N (m is feature dimension, N is feature number)
%   @Dim: dimension of a combination 
%   @Thr: lambda in paper
%
%   output: 
%   @D: sparse combination
for ii = 1 : length(D);
   R(ii).val = D(ii).val*inv(D(ii).val'*D(ii).val)*D(ii).val' - eye(size(D(ii).val,1)); % R matrix in Eq. (13).  
end
save('data/sparse_combinations/D.mat','D');
save('data/sparse_combinations/R.mat','R');
toc;

%% Testing System 

load('data/sparse_combinations/Tw.mat','Tw');
load('data/sparse_combinations/R.mat','R');
ThrTest = 0.20;
ThrMotionVol = 5; 
fileNumAll = 0;
timeAll = 0;
for idx = 1 : testFileNum 
    
    load(['data/testing_vol/vol', sprintf('%.2d',idx), '.mat']); 
    imgVol = im2double(vol);
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
    t2 = toc(t1);
    save(['data/testing_result/regionalRes_',num2str(idx),'.mat'], 'AbEvent3');
    fprintf('we can achieve %d FPS in %d th video \n', round(size(imgVol,3)/t2), idx);
    fileNumAll = fileNumAll + size(imgVol,3);
    timeAll = timeAll + t2;
    
end
fprintf('average FPS is %d \n', round(fileNumAll/timeAll));


%% Accuracy result 
optThr = 0.08;
overlapThr = 0.3;
acc = zeros(1, testFileNum);
for idx = 1 : testFileNum
    
    load(['data/testing_label_mask/', num2str(idx), '_label.mat'], 'volLabel');
    load(['data/testing_result/regionalRes_',num2str(idx),'.mat'], 'AbEvent3');
    ratios = zeros(1, length(volLabel));
    [Hs, Ws] = size(volLabel{1});
    for ii = 1 : length(volLabel)
        curFrameTemp = double(AbEvent3(:,:,ii) > optThr);
        curFrame = boolean(imresize(curFrameTemp ,[Hs, Ws], 'bilinear') > 0);
        unionSet = sum(sum(curFrame|volLabel{ii}));
        interSet = sum(sum(curFrame&volLabel{ii}));
        if unionSet == 0
            ratios(ii) = 1;
        else
            ratios(ii) = interSet/unionSet;
        end
    end
    acc(idx) = sum(ratios > overlapThr)/length(ratios);
    fprintf('Accuracy in %d th video is %.1f %% \n', idx, 100*acc(idx));
end
fprintf('our overall accuracy is %.1f %% \n', 100*mean(acc));  