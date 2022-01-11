type = cell(1,2);
type{1,1} = 'binomial';  type{1,2} = 'probit'; 
group = ones(1, size(Xmis,2));
q= 3; lambda=100; verbose=0;
% imputation using MIG
tic; 
[hX, hD, hHm, hBm] = MIG(Xmis, group, type, q);
time = toc

