function  [feaRawTrain, LocV3Train]  = train_features(fileName, params)
%train_features - "Abnormal Event Detection at 150 FPS in Matlab"
%
%   [feaRawTrain, LocV3Train]  = train_features(fileName, params) extract 3D gradient feature for training video volume 
%
%   input: 
%   @fileName: file name of training video  
%   @params: parameters
%
%   output: 
%   @feaRawTrain: a m x N matrix whose column is m dimemsions 3D gradient feature. (N in total)   
%   @LocV3Train: a 3 x M matrix records 3D spatial-temporal location of 3D
%   gradient feature (N in total).
% 

H = params.H;
W = params.W;
patchWin = params.patchWin;
srs = params.srs;  
trs = params.trs; 
MT_thr = params.MT_thr; 
tprLen = params.tprLen; 
 
load(fileName, 'Video_Output');

for ii = 1 : size(Video_Output, 4)
    Video_Output(:, :, :, ii) = im2double(Video_Output(:, :, :, ii));
    vol(:, :, ii) = rgb2gray(Video_Output(:, :, :, ii));
end
 
voBlur = vol; 
blurKer = fspecial('gaussian', [3,3],1);    %smooth
mask = conv2(ones(H,W), blurKer,'same');    %eliminate the dark border

for pp = 1 : size(vol,3)
    voBlur(:,:,pp) =  conv2(vol(:,:,pp), blurKer, 'same')./mask;
end
volG = abs(voBlur(:,:,1:(end-1)) - voBlur(:,:,2:end)) ;     %caculate gradient of Time

rsNum = 50000; % reserved number:


count = 0;
motionReg = zeros(size(volG)); 
motionResponse = zeros(size(volG));  
for frameID = (tprLen + 1) : ( size(volG, 3) - tprLen)   
    motionReg(:,:,frameID)  = conv2(volG(:,:,frameID), ones(patchWin), 'same'); 
end
 
for frameID = (tprLen + 1) : ( size(volG, 3) - tprLen)       
    motionResponse(:,:,frameID)  =  sum(motionReg(:,:,frameID-2:frameID+2),3);  %accumulate the adjacent 5 frames
end



feaRawTrain = zeros(tprLen*patchWin^2, rsNum);
LocV3Train  = zeros(3, rsNum); 
    
    
for frameID = (tprLen + 1) : trs : ( size(volG, 3) - tprLen)   
    for ii = round(patchWin/2)+1 : srs : H - round(patchWin/2)
        for jj = round(patchWin/2)+1 : srs : W - round(patchWin/2) 
            if motionResponse(ii,jj,frameID) > MT_thr
                count = count + 1;
                cube = volG(ii - round(patchWin/2):ii + round(patchWin/2)-1, jj - round(patchWin/2):jj + round(patchWin/2)-1, frameID-2:frameID+2);
                feaRawTrain(:,count) = cube(:);             %expand by column
                LocV3Train(:,count) =  [ii;jj;frameID]'; 
            end
        end
    end 
end

delIdx = find(sum(LocV3Train) == 0);
feaRawTrain(:,delIdx) = [];         %take out all-zero columns
LocV3Train(:,delIdx) = [];

feaRawTrain = bsxfun(@rdivide, feaRawTrain, sqrt(sum(feaRawTrain.^2)));

end