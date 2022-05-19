clear
clc

addpath("./knn-graphs-1.0/")

%% select a dataset
Hc = load("./Datasets/UCI_data/preprocessed/heart-disease.classes");
Hf = load("./Datasets/UCI_data/preprocessed/heart-disease.features");
% Hc = load("./Datasets/UCI_data/preprocessed/iris.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/iris.features");
% Hc = load("./Datasets/UCI_data/preprocessed/wine.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/wine.features");
% Hc = load("./Datasets/UCI_data/preprocessed/breast-cancer.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/breast-cancer.features");
% Hc = load("./Datasets/UCI_data/preprocessed/car.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/car.features");
% Hc = load("./Datasets/UCI_data/preprocessed/winequality.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/winequality.features");
% Hc = load("./Datasets/UCI_data/preprocessed/internet-ads.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/internet-ads.features");
% Hc = load("./Datasets/UCI_data/preprocessed/yeast.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/yeast.features");
% Hc = load("./Datasets/UCI_data/preprocessed/abalone.classes");
% Hf = load("./Datasets/UCI_data/preprocessed/abalone.features");

Hfn = normalize(Hf);
n = length(Hf);

G = knngraph(Hfn, 5);
At = zeros(n);
Edges = G.Edges{:,:};
ind = sub2ind([n,n],Edges(:,1),Edges(:,2));
At(ind)=1.;
A = double((At + At')>0);
Deg = diag(sum(A));
L = Deg - A;
Dn = diag(diag(Deg).^(-0.5));
%L = diag(diag(Deg).^(-0.5))*L*diag((diag(Deg).^(-0.5)));

k = max(Hc)+1;
%% minimum cut
[V,D] = eig(L);
Fvec = V(:,1:k);


%% true label
Xtrue = zeros(n,k);
for ii = 1:k
    Xtrue(find(Hc==ii-1),ii) = 1;
end

%% 20% obs labels
nr = round(n*0.2);
y = zeros(n,k);
sind = randperm(n,nr);
tmp1 = zeros(n,1);
tmp1(sind,1) = 1;
M = diag(tmp1);
y = M*Xtrue;
%y(sind,:) = Xtrue(sind,:);
%y = normalize(y);


%% prior belief
R = zeros(n,k);
R(:) = 1/k;


%% data
dlen = length(y);

%% parameters
k = k;
Lambda = 0.01;
esp = 0.01;
ItrNum = 100;
nmax = 1200;
percent_seed = 0.2;


%% test 10 times
MissRate = [];
for i = 1 : 10
    %% 20% obs labels per class
    sind = [];
    for j = 1 : k
        tind = find(Xtrue(:,j)==1);
        tnum = ceil(length(tind)*percent_seed);
        sind = [sind; tind(randperm(length(tind), tnum))];
    end
    y = zeros(n,k);
    tmp1 = zeros(n,1);
    tmp1(sind,1) = 1;
    M = diag(tmp1);
    y = M*Xtrue;
    
    X=y;
    Xresult = GFLL0_ssl_k0_FW(y, L, X, M, R, Lambda, eps, ItrNum);
    MissRate = [MissRate sum(sum(Xresult.*(1- Xtrue))) / n]
end





%% local function

function Xresult = GFLL0_ssl_k0_FW(y, L, X, M, R, Lambda, eps, ItrNum)

%% Frank-wolf v1
dlen = length(y);
k = size(y,2);
X0 = X;
ObjListv1 = [];
for t = 1 : ItrNum
    Xt = X0;
    ObjListv1 = [ObjListv1 GFLL0_ssl_Obj(y, L, Xt, M, R, Lambda, eps)];
    
    % linear step
    Dev = GFLL0_ssl_Dev(y, L, Xt, M, R, Lambda, eps);
    %     DevCopy = repmat(Dev, 1, nmax);
    %     M = matchpairs(DevCopy,1);
    %     Direction = zeros(dlen, k);
    %     for ii = 1 : length(M)
    %         id1 = M(ii,1);
    %         id2 = mod(M(ii,2)-1,k)+1;
    %         Direction(id1, id2) = 1;
    %     end
    [C,Indc] = min(Dev,[],2);
    Indr = (1:dlen)';
    Lind = sub2ind(size(Dev), Indr, Indc);
    Direction = zeros(dlen, k);
    Direction(Lind) = 1;
    
    % step size search
    objbest = 1e10;
    stbest = 0;
    objtmplist = [];
    for stt = 0.1:0.1:1
        st = (2/(t+2))*stt;
        objtmp = GFLL0_ssl_Obj(y, L, Xt+st*(Direction-Xt), M, R, Lambda, eps);
        objtmplist = [objtmplist objtmp];
        if objtmp < objbest
            objbest = objtmp;
            stbest = st;
        end
    end
    
    % update
    X0 = Xt+stbest*(Direction-Xt);
    %     [C,Indc] = max(X0t,[],2);
    %     Indr = (1:dlen)';
    %     Lind = sub2ind(size(X0), Indr, Indc);
    %     X0 = zeros(dlen, k);
    %     X0(Lind) = 1;
    %     X0tCopy = repmat(X0t, 1, nmax);
    %     M = matchpairs(X0tCopy,1);
    %     X0 = zeros(dlen, k);
    %     for ii = 1 : length(M)
    %         id1 = M(ii,1);
    %         id2 = mod(M(ii,2)-1,k)+1;
    %         X0(id1, id2) = 1;
    %     end
end

[C,Indc] = max(X0,[],2);
Indr = (1:dlen)';
Lind = sub2ind(size(X0), Indr, Indc);
Xresult = zeros(dlen, k);
Xresult(Lind) = 1;
end

function deriv = GFLL0_ssl_Dev(y, L, X, M, R, Lambda, eps)
n = length(y);
I = eye(n);
MatD = -2*(M*y+eps*R)*y'*M;
Fi = inv(X'*(M+I)*X);
Dev1 = (I - (M+I)*X*Fi*X')*(MatD+MatD')*X*Fi;
MatD2 = -2*eps*(M*y+eps*R)*R';
Dev2 = (I - (M+I)*X*Fi*X')*(MatD2+MatD2')*X*Fi;
F = M*y+eps*R;
Dev3 = 2*(I - (M+I)*X*Fi*X')*(F*F'*X*Fi*X'*M + M*X*Fi*X'*F*F')*X*Fi;
Dev4 = 2*(I - (I+I)*X*Fi*X')*(F*F'*X*Fi*X'*I + I*X*Fi*X'*F*F')*X*Fi;
deriv = Dev1 + Dev2 + Dev3 + Dev4 + 2*Lambda*L*X;
end

function obj = GFLL0_ssl_Obj(y, L, X, M, R, Lambda, eps)
Mat1 = M*(y-X);
Mat2 = X'*L*X;
Mat3 = R-X;
obj = norm(Mat1,"fro")^2 + eps*norm(Mat3,"fro")^2 + Lambda*trace(Mat2);
end


