function [feaPCA, LocV3] = test_features(feaVol, Tw, ThrMotionVol, params)  
%train_features - "Abnormal Event Detection at 150 FPS in Matlab"
%
%   [feaPCA, LocV3] = test_features(feaVol, Tw, ThrMotionVol, params)
%   extract 3D gradient feature for training video volume
%
%   input: 
%   @feaVol: 3D gradient volumn
%   @Tw:     PCA compression matrix
%   @ThrMotionVol:  motion volume threshold
%   @params: parameter set
%
%   output: 
%   @feaPCA: PCA feature 
%   @LocV3:  selected 3D gradient volume location 
%

    patchWin = params.patchWin; 
    tprLen = params.tprLen;
    BKH = params.BKH;
    BKW = params.BKW;  
    rsNum = 30000; % reserved number:
    hftprLen = (tprLen - 1)/2;

    count = 0;
    motionReg = zeros(BKH, BKW, size(feaVol,3)); % motion region
 
    for ii = 1 : BKH
        for jj = 1 : BKW
             motionReg(ii,jj,:)= sum(sum(feaVol(1+patchWin*(ii-1):patchWin*ii, 1+patchWin*(jj-1):patchWin*jj, :), 2),1);
        end
    end
  
    feaRaw = zeros(tprLen*patchWin^2, rsNum);
    LocV3  = zeros(3, rsNum); 
    
    
    for frameID = (tprLen + 1) : ( size(feaVol, 3) - tprLen)   
        
        motionVol  = sum(motionReg(:,:,frameID-2:frameID+2),3); 

        for ii = 1 : BKH
            for jj = 1 : BKW 
                if motionVol(ii,jj) > ThrMotionVol
                    count = count + 1;
                    cube = feaVol(1+patchWin*(ii-1):patchWin*ii, 1+patchWin*(jj-1):patchWin*jj, frameID-hftprLen:frameID+hftprLen);
                    feaRaw(:,count) = cube(:);
                    LocV3(:,count) =  [ii;jj;frameID]'; 
                end
            end
        end 
    end
    delIdx = find(sum(LocV3) == 0);
    feaRaw(:,delIdx) = [];
    LocV3(:,delIdx) = [];
    
    feaRaw = bsxfun(@rdivide, feaRaw, sqrt(sum(feaRaw.^2)));
   
    feaPCA = Tw*feaRaw;

end


