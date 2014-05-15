%% Parameters 
set_params;

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
optThr = 0.12;
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