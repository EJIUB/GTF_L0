clc
clear


%% get data
load ./Datasets/Community_0.2_0.05.mat
% load Community_0.05_0.01.mat % anyother graph

n=length(y_true);
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
yr = round(y_true);
%y = yr - mean(yr);
dlen = length(yr);



%% SNR and parameters
INPUT_SNR = -15:5:30;
k = 3;
Lambda = [1, 10, 100, 1000,10000,100000];
ItrNum = 100;
Numtrail = 10;
d = 10;
Nmin = 20;

Xopt = zeros(dlen,k);
idx = kmeans(yr,k);
idy = (1:dlen)';
Lind = sub2ind([dlen,k], idy, idx);
Xopt = zeros(dlen, k);
Xopt(Lind) = 1;


y_true_norm_sq = norm(yr)^2;
SIGMA_SQ = (y_true_norm_sq./ 10.^(INPUT_SNR ./ 10.0) )/ dlen;

mean_snr = zeros(Numtrail, length(INPUT_SNR));
ObjOpt = zeros(Numtrail, length(INPUT_SNR));
ObjRes = zeros(Numtrail, length(INPUT_SNR));
for tt = 1 : Numtrail
    
    for ss = 1 : length(SIGMA_SQ)
        
        for zz = 1 : length(Lambda)
            %% generate y
            yn = repmat(yr,1,d) + sqrt(SIGMA_SQ(ss)).*randn(dlen,d);
            ymean = repmat(mean(yn),dlen,1);
            y = yn - ymean;
            
            %% initilization
            X = zeros(dlen, k);
            
            idx = kmeans(y,k);
            idy = (1:dlen)';
            Lind = sub2ind(size(X), idy, idx);
            Xinit = zeros(dlen, k);
            Xinit(Lind) = 1;
            ObjInit =  GFLL0_Obj(y, L, Xinit, Lambda(zz));
            ObjOpt(tt,ss) =  GFLL0_Obj(y, L, Xopt, Lambda(zz));
            
            %[Xinit] = GFFL0_Spectral(y, L, X, Lambda);
            X = Xinit;
            
            %% call FW
            [Xresult] = GFFL0_Spectral_v2(y, L, X, Lambda(zz));
            ObjRes(tt,ss,zz) = GFLL0_Obj(y, L, Xresult, Lambda(zz));
            
            %% Get Beta
            beta_hat = Xresult*inv(Xresult'*Xresult)*Xresult'*yn;
            
            %% SNR outp
            ses = vecnorm(beta_hat-yr).^2;
            ouput_snr = 10*log10(y_true_norm_sq./ses);
            mse = mean(ses);
            mean_snr(tt,ss,zz) = 10*log10(y_true_norm_sq/mse);
            
        end
        
        
    end
    
    
    
    
end

% % %% result
L0_vec = max(mean_snr,[],3);
plot(L0_vec);


%% local function


function [Xresult] = GFFL0_Spectral(y, L, X, Lambda)
[n,k] = size(X);
M = -y*y' + Lambda * L;
[V, D] = eig(M);
ev = diag(D);
[sev, ind] = sort(ev);
idx = kmeans(V(:,ind(1:k)),k);
idy = (1:n)';
Lind = sub2ind(size(X), idy, idx);
Xresult = zeros(n, k);
Xresult(Lind) = 1;
end

function [Xresult] = GFFL0_Spectral_v2(y, L, X, Lambda)
[n,k] = size(X);
M = y*y';
[V, D] = eig(M);
ev = diag(D);
[sev, ind] = sort(-ev);
ScaleFactor = ev(ind(1:k)).^(0.5);
VE1 = V(:,ind(1:k))*diag(ScaleFactor);

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

function [Xresult]  = GFFL0_FW_v2(y, L, X, Lambda, Nmin1, ItrNum)
%% Frank-wolf v1
Nmin = 10;
X0 = X;
dlen = length(y);
k = size(X,2);
ObjListv1 = [];
ObjDis= [];
ObjDis_Best = 1e10;
BestDirect = zeros(dlen,k);
ObjOpt = GFLL0_Obj(y, L, X, Lambda);
for t = 1 : ItrNum
    Xt = X0;
    ObjListv1 = [ObjListv1 GFLL0_Obj(y, L, Xt, Lambda)];
    
    % linear step
    Dev = GFLL0_Dev(y, L, Xt, Lambda);
    %     DevCopy = repmat(Dev, 1, nmax);
    %     M = matchpairs(DevCopy,1);
    %     Direction = zeros(dlen, k);
    %     for ii = 1 : length(M)
    %         id1 = M(ii,1);
    %         id2 = mod(M(ii,2)-1,k)+1;
    %         Direction(id1, id2) = 1;
    %     end
    DirectionT = ProjAssignment_Gruobi(Dev, Nmin1);
    Direction = reshape(DirectionT, dlen, k);
    ObjDis= [ObjDis GFLL0_Obj(y, L, Direction, Lambda)];
    if ObjDis_Best > ObjDis(end)
        ObjDis_Best = ObjDis(end);
        BestDirect = Direction;
    end
    
    % step size search
    objbest = 1e10;
    stbest = 0;
    objtmplist = [];
    for stt = 0.1:0.2:1
        st = (2/(t+2))*stt;
        objtmp = GFLL0_Obj(y, L, Xt+st*(Direction-Xt), Lambda);
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
%     Lind = sub2ind(size(X0t), Indr, Indc);
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

function [Xresult]  = GFFL0_FW(y, L, X, Lambda, ItrNum)
%% Frank-wolf v1
X0 = X;
dlen = length(y);
k = size(X,2);
ObjListv1 = [];
for t = 1 : ItrNum
    Xt = X0;
    ObjListv1 = [ObjListv1 GFLL0_Obj(y, L, Xt, Lambda)];
    
    % linear step
    Dev = GFLL0_Dev(y, L, Xt, Lambda);
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
    for stt = 0.1:0.2:1
        st = (2/(t+2))*stt;
        objtmp = GFLL0_Obj(y, L, Xt+st*(Direction-Xt), Lambda);
        objtmplist = [objtmplist objtmp];
        if objtmp < objbest
            objbest = objtmp;
            stbest = st;
        end
    end
    
    % update
    X0 = Xt+stbest*(Direction-Xt);
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

function [up] = ProjAssignment_Gruobi(U,Nmin)

%% test projection (https://www.gurobi.com/documentation/9.5/refman/matlab_api_overview.html#matlab:MATLAB)
% n = length(M);
% H = eye(n);
%f = u;
% 
[n,k] = size(U);
f = reshape(U, n*k, 1);
At = zeros(n+k, k*n);
for ii = 1 : n
    for jj = 0 : k-1
        At(ii, ii+(n*jj)) = 1;
    end
end
for ii = 0 : k-1
    At(ii+n+1,ii*n+1:ii*n+n) = 1;
end
ub = ones(k*n,1);

% Set objective: x^THx + 2f^Tx
model.obj = f;
% model.Q = sparse(H);

model.A = sparse(At);
model.rhs = [ones(n,1);repmat(Nmin,[k,1])];

% Add quardic constraint: x^TMx<=k
% model.quadcon(1).Qc = sparse(M);
% model.quadcon(1).q  = zeros(n,1);
% model.quadcon(1).rhs = k;

% Add bounds for x: 0<=x<=1
% model.ub = ub;

model.sense = strcat(repmat('=',[1,200]),'>>>');
model.vtype = 'B';

params.outputflag = 0;
output = gurobi(model,params);

up = output.x;
end



function [up] = MinCut_Gruobi(U,L,Nmin)

%% test projection (https://www.gurobi.com/documentation/9.5/refman/matlab_api_overview.html#matlab:MATLAB)
% n = length(M);
% H = eye(n);
%f = u;
% 
[n,k] = size(U);
f = reshape(U, n*k, 1);
At = zeros(n+k, k*n);
for ii = 1 : n
    for jj = 0 : k-1
        At(ii, ii+(n*jj)) = 1;
    end
end
for ii = 0 : k-1
    At(ii+n+1,ii*n+1:ii*n+n) = 1;
end
ub = ones(k*n,1);

% Set objective: x^THx + 2f^Tx
%model.obj = f;
% model.Q = sparse(H);
Lk = L;
for ii = 1 : k-1
    Lk = blkdiag(Lk,L);
end
model.Q = sparse(Lk);

model.A = sparse(At);
model.rhs = [ones(n,1);repmat(Nmin,[k,1])];

% Add quardic constraint: x^TMx<=k
% model.quadcon(1).Qc = sparse(M);
% model.quadcon(1).q  = zeros(n,1);
% model.quadcon(1).rhs = k;

% Add bounds for x: 0<=x<=1
% model.ub = ub;

model.sense = strcat(repmat('=',[1,n]),'>>>');
model.vtype = 'B';

params.outputflag = 0;
output = gurobi(model,params);

up = output.x;
end
