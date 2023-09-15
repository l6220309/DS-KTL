clc;
clear all;
close all;
warning off;

% Load datasets: 
% 9 subjects, each 22*750*144 (channels*points*trails)
root='MI2-6\';
listing=dir([root '*.mat']);
addpath('lib');

% Load data and perform congruent transform
fnum=length(listing);
Ca=nan(22,22,144*fnum);
Xr=nan(22,750,144*9);
Xa=nan(22,750,144*9);
Y=nan(144*fnum,1);
ref={'riemann','logeuclid','euclid'};
for f=1:fnum
    load([root listing(f).name])
    idf=(f-1)*144+1:f*144;
    Y(idf) = y; Xr(:,:,idf) = x;
    Ca(:,:,idf) = centroid_align(x,ref{2});
%     [~,Xa(:,:,idf)] = centroid_align(x,ref{3});
end
tic;
All_acc=[];
All_std=[];
for t = 1:253
    t
    BCA=zeros(fnum,1);
    bca_dte = [];
    std_dte = [];
    for n=1:fnum
%         disp(n)
        % Single target data & multi source data
        idt=(n-1)*144+1:n*144;
        ids=1:144*fnum; ids(idt)=[];             
        Yt=Y(idt); Ys=Y(ids);
        idsP=Yt==1; idsN=Yt==0;
        Ct=Ca(:,:,idt);  Cs=Ca(:,:,ids);

        % Logarithmic mapping on aligned covariance matrices
        Xs=logmap(Cs,'MI'); % dimension: 253*1152 (features*samples)
        Xt=logmap(Ct,'MI');
        
        options.p = 200;
        options.mu = 0.01; 
        options.iter = 5;
        options.d = t;             % subspace bases 
        options.alpha= 0.01;        % the parameter for source discriminability
        options.beta = 0.1;         % the parameter for target locality, default=0.1
        options.rho = 20;           % the parameter for subspace discrepancy
        
         % MDFS feature selection
        para.alpha = 1; para.beta = 1; para.gamma = 100; para.k = 0;
        train_f_size = size(Xs);
        [W, obj] = MDFS(Xs', Ys, para);
        [dumb, idx] = sort(sum(W.*W,2),'descend'); 
        feature_idx = idx(1:train_f_size(1));
        index = feature_idx(1:t);  % L vs. R:180, 78.01%   F vs. T :150 70.14%
        % 2-2 180, 2-3 150, 2-4
        Xss = Xs(index,:);
        Xts = Xt(index,:);
        Ys = Ys + 1;
        Yt = Yt + 1;

        [result,mu]=MEICSC(Xss,Ys,Xts,Yt,options);
        BCA = result(end);


        bca_dte=[bca_dte,mean(BCA)*100];
    end
    toc
%     disp(mean(bca_dte));
%     disp(sqrt(var(bca_dte)));
    acc = mean(bca_dte);
    std = sqrt(var(bca_dte));
    All_acc=[All_acc,acc];
    All_std=[All_std,std];
end

rmpath('lib');
