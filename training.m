%% Parameters 
params.H = 90;       % loaded video height size
params.W = 160;       % loaded video width size
params.patchWin = 10; % 3D patch spatial size 
params.tprLen = 5;    % 3D patch temporal length
params.BKH = 9;      % region number in height
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
for ii = 1 : 1%length(trainVolDirs)
    [feaRawTrain, LocV3Train]  = train_features('data/WD_training.mat', params);%train_features([fileName,'/', trainVolDirs{ii}], params);
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
D = sparse_combination(feaMatPCA, 20, 0.22);
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