%% Parameters 
set_param;

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
numEachVol = 100000; % The maximum sample number in each training video is 7000 
trainVolDirs = name_filtering(fileName); 
Cmatrix = zeros(tprLen*patchWin^2, 6 * numEachVol);
rand('state', 0);
for ii = 1 : 6
    [feaRawTrain, LocV3Train]  = train_features(['data/CV_Abnormality_New_Training_', num2str(ii), '.mat'], params);
    t = randperm(size(feaRawTrain,2));
    curFeaNum = min(size(feaRawTrain,2),numEachVol);
    Cmatrix(:, numEachVol*(ii - 1) + 1 : numEachVol*(ii - 1) + curFeaNum) =  feaRawTrain(:,t(1:curFeaNum));     %put random curFeaNum column into Cmatrix
    disp(['Feature extraction in video ', num2str(ii),' is done!'])
end
Cmatrix(:,sum(abs(Cmatrix)) == 0) = [];         %take out the zero valued columns

COEFF = princomp(Cmatrix');                     %compress raws
Tw = COEFF(:,1:PCAdim)';
feaMatPCA = Tw*Cmatrix;  
save('data/sparse_combinations/Tw.mat','Tw');
toc;

%% Sparse combination learning  (about 4 minutes)
tic;
%D = sparse_combination(feaMatPCA, 25, 0.22);
D = sparse_combination_old(feaMatPCA, 20, 0.10);
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