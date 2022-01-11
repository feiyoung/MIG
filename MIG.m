function [hX, hD, hHm, hBm,cvVals, history] = MIG(Xmiss, group, type, q, epsLogLik,...
    maxIter,lambda, output, parallel)

%--------------------------------------------------------------------------
% gfmImpute.m:  Missing value Imputation by Generalized factor models (MIG).
%--------------------------------------------------------------------------
% The function to conduct the Missing value Imputation by Generalized
% factor models (MIG).
%----------------------------------------------------------------------------------
% Author:       Liu Wei
% Maintainer:    Liu Wei <weiliu@smail.swufe.edu.cn>
% Date:         June. 25, 2021
% Copyright (c) 2021, Liu Wei
% All rights reserved.
%
% DESCRIPTION:
%    Missing value Imputation via Generalized factor model. It is  
%    scalable for high-dimensional large-scale  mixed-type data.
%
% USAGE:
%    hX = MIG(Xmiss, group, type, q)
%    [hX, hD, hHm, hBm,cvVals, history] = MIG(Xmiss, group, type, q, epsLogLik,...
%    maxIter,lambda, output, parallel)
%    (Use empty matrix [] to apply the default value, eg. hX = MIG(Xmis,
%    group, type, q,[]);
%
%
% INPUT ARGUMENTS:
% Xmiss        Input missing matrix, of dimension nobs x nvars; each row is an
%             observation vector, where missing values are NaNs. 
%             n*p(p=(p1+p2+..+p_d)),observational mixed data matrix, d is the types of variables,
%             p_j is the dimension of j-th type variable.
% group       a vector with length equal to p, specify each column of X belonging to which group.
% type        a d*2 cell, specify the type of variables and the link function in each group. 
%             For example, type={'poisson', 'log';  'binomial', 'logit')
% q           a positive integer or empty, specify the number of factors. 
%             IC criteria is used to dertemined $q$ by using selectFacNumber.m function.
%             
% OUTPUT ARGUMENTS:
% hX          an imputed matrix.
% hHm         a n*(q+1) matrix, the estimated factor matrix plus a column of ones.
% hBm         a p*(q+1) matrix, the estimated loading-intercept matrix.

if(~exist('epsLogLik', 'var') || isempty(epsLogLik))
    epsLogLik = 1e-4; % A ridge penalty is added to increase the stabality of algorithm.
end 

if(~exist('maxIter', 'var') || isempty(maxIter))
    maxIter = 10; % A ridge penalty is added to increase the stabality of algorithm.
end 

if(~exist('lambda', 'var') || isempty(lambda))
    lambda = 0; % A ridge penalty is added to increase the stabality of algorithm.
end
if(~exist('output', 'var'))
    output = 0;
end
if(~exist('parallel', 'var') || isempty(parallel))
    parallel = 0;
end
[Xmis,Xmiss, group, type, type_scale] = precheckdata(Xmiss, group, type);

ind_set = unique(group);
ng = length(ind_set);


gcell = cell(1, ng);
if parallel
    parfor j = 1:ng
        gcell{j} = find(group==j);
    end
else
    for j = 1:ng
    gcell{j} = find(group==j);
    end
end
[n,p] = size(Xmis);
% initialize
warning('off');
hmu = mean(Xmis, 'omitnan');
Xm = Xmis - repmat(hmu, n, 1); Xm(isnan(Xm))=0;
[hH, hB] = factorm(Xm, q, 1);
clear Xm;
eps1 = 1e-4;
hHm = [ones(n,1) hH];
hBm = [hmu' hB];
% hHm(1:4, 1:4), hBm(1:4,1:4)
dBm = Inf; dH = Inf; dOmega = Inf;dc =Inf;
dOmega = max([dBm, dH]);
tmpBm = zeros(p,q+1);tmpHm = hHm; tmpc = 1e7; dc = 1e7;
% maxIter = 50;
k = 1;
tic;
while k <= maxIter && dOmega > eps1 && dc > epsLogLik
    
    hhB = [];
    for jj = 1:ng
        % jj = 2
        [B1] = localupdateB2(Xmis(:,gcell{jj}), hH, type(jj,:), parallel);
        % [B1] = localupdateB3(Xmis, gcell{j}, hHm, type(j,:));
        hhB = [hhB, B1];
    end
    hBm = hhB';
   
%     hmu = hBm(:,1);
%     hB = hBm(:,2:end);
%     % ensure indentifiability.
%     [B0tmp, ~] = qr(hB, 0);
%     B0= B0tmp * diag(sort(sqrt(eig(hB'*hB)), 'descend'));
%     
%     sB = sign(B0(1,:));
%     hB = B0.*repmat(sB,p,1); % ensure B first nonzero is positive
%     %hB(1:4,:), B(1:4,:)
%     hBm = [hmu, hB];
    dB = norm(hBm - tmpBm, 'fro')/norm(hBm, 'fro');
    tmpBm = hBm;
    % given B^(1), update H^(1)
    % tmp = updateH(Xmis, hBm, hHm, gcell, type);
    % tmp(1:6,:)
    hHm = updateH(Xmis, hBm, hHm, gcell, type, lambda, parallel);
    % hHm(1:5, 1:4)
     
    hH = hHm(:,2:end);
%     hH0 = hHm(:,2:end);
%     [H0, ~] = qr(hH0, 0);
%     hH1 = H0 * sqrt(n);
%     sH0 = sign(hH0(1,:)).* sign(hH1(1,:));
%     hH = hH1.* repmat(sH0,n,1);
%     hHm = [ones(n,1), hH];
    dH = norm(hHm-tmpHm, 'fro')/norm(hHm, 'fro');
    tmpHm = hHm;
    dOmega = max([dB, dH]);
    c = objMisfun(hHm, hBm, Xmis, gcell, type);
    dc = abs(c - tmpc)/abs(tmpc);
    tmpc = c;
    if output
        fprintf('Iter %d \n', k);
        fprintf('dH= %4f,dc= %4f, c=%4f \n', dH,dc, c);
    end
    history.dOmega(k) = dOmega; history.dc(k)=dc; history.c(k)=c;history.realIter = k;
    k = k+1;
end

hhB = [];
for j = 1:ng
        
        [B1] = localupdateB2(Xmis(:,gcell{j}), hH, type(j,:), parallel);
        hhB = [hhB, B1];
end
hBm = hhB';
history.maxIter = maxIter;

% compute criteria values
cvVals =[log(c+1e-12)+ q*(n+p)/(n*p)*log(n*p/(n+p)), ...
               %IC"=sum(log(c1+1e-12), q/min(sqrt(n), sqrt(p))^2*log(min(sqrt(n), sqrt(p))^2)),
               c + q * (n+p)/(n*p)*log(n*p/(n+p))];


hD = hHm * hBm';
hmu = mean(hD);
tD = hD - repmat(hmu, n, 1);
[U,S,V] = svds(tD,q);
hHm = [ones(n,1), sqrt(n)* U];
hBm = [hmu', V*S/sqrt(n)];
hX = Xmis;
for j= 1:ng
    switch type{j,1}
    case 'normal'
       hX(:, gcell{j}) = hD(:, gcell{j});
       smat = type_scale{j,2};
       if(~isempty(smat))
           hX(:, smat(1,:)) = hX(:, smat(1,:)) .* repmat(smat(2,:), n, 1);
       end
    case 'poisson'
       hX(:, gcell{j}) = round(exp(hD(:, gcell{j})));
       smat = type_scale{j,2};
       if(~isempty(smat))
           hX(:, smat(1,:)) = hX(:, smat(1,:)) .* repmat(smat(2,:), n, 1);
       end
    case 'binomial'
       hX(:, gcell{j}) = (1 ./ (exp(-hD(:, gcell{j})) +1) > 0.5) + 0;
    end
end
O = ~isnan(Xmiss);
hX(O) = Xmiss(O);