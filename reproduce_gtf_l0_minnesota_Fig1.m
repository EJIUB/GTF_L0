clc
clear


%% get data
load ./Datasets/Minnesota.mat
n=length(y_noise);
[D,edges]=preprocess_D(Dx,n);
base=norm(D,1);

%% edge to adj matrix
At = zeros(n,n);
ind = sub2ind([n,n],edges(:,1),edges(:,2));
At(ind)=1;
A = At + At';
Deg = diag(sum(A));
L = Deg - A;


%% data
yr = y_noise';
y = yr - mean(yr);
dlen = length(y);

%% parameters
k = 4;
Lambda = 1;
ItrNum = 100;
nmax = 1200;

%% spectral
X = zeros(dlen,k);
[Xresult] = GFFL0_Spectral_v2(y, L, X, Lambda);




%% Get Beta
beta = Xresult*inv(Xresult'*Xresult)*Xresult'*y+mean(yr);

% MSE
MSE = norm(beta-double(beta_0)')^2/n;

% TPR and FPR
beta0 = double(beta_0)';
t0 = Dx*beta0;
t0(abs(t0) < 0.1) = 0;
t0supp = t0 > 0;

t = Dx*(beta);
t(abs(t) < 0.1) = 0;
tsupp = t > 0;

total = length(beta);
TP = sum(t0supp.*tsupp); % true positive
FN = sum(t0supp.*(1- tsupp)); % false negative
FP = sum((1-t0supp).*tsupp); % false positive
TN = total - TP - FN - FP;

TPR = double(TP)/(TP+FN)
FPR = double(FP)/(FP+TN)


%% local function
function [Xresult] = GFFL0_Spectral_v2(y, L, X, Lambda)
[n,k] = size(X);
M = y*y';
[V, D] = eigs(M,k,'largestreal');
ev = diag(D);
ScaleFactor = ev.^(0.5);
VE1 = V*diag(ScaleFactor);

[Vl, Dl] = eig(L);
evl = diag(Dl);
[sevl, indl] = sort(evl);
c = sum(evl(indl(k+1:end))) / (n-k);
ScaleFactorL = sqrt(Lambda)*(c - evl(indl(1:k))).^(0.5);
VE2 = Vl(:,indl(1:k))*diag(ScaleFactorL);

VE = [VE1, VE2];
idx = kmeans(VE,k);
idy = (1:n)';
Lind = sub2ind(size(X), idy, idx);
Xresult = zeros(n, k);
Xresult(Lind) = 1;
end



function deriv = GFLL0_Dev(y, L, X, Lambda)
I = eye(length(y));
Dev1 = (I-X*inv(X'*X)*X')*y*y'*X*inv(X'*X);
Dev2 = L*X;
deriv = -Dev1 + Lambda*Dev2;
end

function obj = GFLL0_Obj(y, L, X, Lambda)
Mat1 = y*y'*X*inv(X'*X)*X';
Mat2 = X'*L*X;
obj = -0.5*trace(Mat1) + 0.5*Lambda*trace(Mat2);
end
