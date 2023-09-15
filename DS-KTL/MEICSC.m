function [result,mu]= MEICSC(Xs,Ys,Xt,Yt,options)
mu=[];
if ~isfield(options,'iter')
    options.iter = 10;
end
if ~isfield(options,'p')
    options.p = size(Xs,1);
end
dim=options.p;
if dim>size(Xs,1)
     options.p = size(Xs,1);
end
T = options.iter;
C = length(unique(Ys));
result=[];
pseudoLabels=[];
cur = 0.0;
for iter = 1:T
    % solving (9)
    [Zs,Zt] = JPDA(Xs,Xt,Ys,pseudoLabels,options);
    
    %% ÌØÕ÷·Ö²¼Í¼
%     fea1 = tsne(Zs);
%     fea2 = tsne(Zt);
% 
%     sour1 = fea1(Ys==1,:);
%     sour2 = fea1(Ys==2,:);
%     targ1 = fea2(Yt==1,:);
%     targ2 = fea2(Yt==2,:);
% 
%     h1 = scatter(sour1(:,1),sour1(:,2),25,[0.3569 0.8588 0.6941], 'filled');
%     hold on
%     h2 = scatter(sour2(:,1),sour2(:,2),25, [0.4196 0.8706 0.8980], 'filled');
%     hold on
%     h3 = scatter(targ1(:,1),targ1(:,2),25,[0.9922 0.6078 0.1569], 'filled');
%     hold on
%     h4 = scatter(targ2(:,1),targ2(:,2),25,'r', 'filled');
%     xlabel('feature 1');
%     ylabel('feature 2');
%     grid on
%     legend('Source Class 1','Source Class 2','Target Class 1','Target Class 2');
        %%

    % solving (5)
    Zmean = mean([Zs;Zt]);
    Zs = Zs - repmat(Zmean,[size(Zs,1) 1 ]);
    Zt = Zt - repmat(Zmean,[size(Zt,1) 1 ]);
    Zs = L2Norm(Zs);
    Zt = L2Norm(Zt);
    %% distance to class means
    classMeans = zeros(C,size(Zs,2));
    for i = 1:C
        classMeans(i,:) = mean(Zs(Ys==i,:));
    end
    % solving (6)
    classMeans = L2Norm(classMeans);
    distClassMeans = EuDist2(Zt,classMeans);
    expMatrix = exp(-distClassMeans);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 C]);
    % solving (7)
    [prob,predLabels] = max(probMatrix');
    % Definition 1
    selelctPercet=iter/T;
    p=(1-selelctPercet);   %*(1-options.selective);
    [sortedProb,index] = sort(prob);
    sortedPredLabels = predLabels(index);
    trustable = zeros(1,length(prob));
    for i = 1:C
        thisClassProb = sortedProb(sortedPredLabels==i);
        if ~isempty(thisClassProb)
            trustable = trustable+ (prob>thisClassProb(floor(length(thisClassProb)*p)+1)).*(predLabels==i);
        end
    end
    % Definition 2
    pseudoLabels = predLabels;
    pseudoLabels(~trustable) = -1;
    % calculate ACC
    acc = sum(predLabels'==Yt)/length(Yt);
    if acc > cur
        cur = acc;
        result=[result,acc];
    else
        return
    end
    
    % solving (10)-(12)
    options.mu = ACDA(Zs,Zt,Ys,pseudoLabels',C);
    mu=[mu,options.mu];
   % fprintf('Iteration=%d,mu=%0.3f, Acc:%0.3f\n', iter,options.mu, acc);
end
end



function [Zs,Zt] = JPDA(Xs,Xt,Ys,YtPseudo,options)

mu = options.mu;
% gamma = options.gamma;
d = options.d;
alpha = options.alpha; 
beta = options.beta;
rho = options.rho; 



% Set variables
% X = [Xs,Xt];
% X = X*diag(sparse(1./sqrt(sum(X.^2))));
% [m,n] = size(X);
[ms, ns] = size(Xs);
[mt, nt] = size(Xt);
class = unique(Ys);
C = length(class);

% Initialize P: source domain discriminability
meanTotal = mean(Xs,2);
Sw = zeros(ms);
Sb = zeros(ms);
for i=1:C
    Xi = Xs(:,Ys==class(i));
    meanClass = mean(Xi,2);
    Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
    Sw = Sw + Xi*Hi*Xi';
    Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
end
P = zeros(2*ms,2*ms); P(1:ms,1:ms) = Sw;
P0 = zeros(2*ms,2*ms); P0(1:ms,1:ms) = Sb;

% Initialize L: target data locality
manifold.k = 10; % default set to 10
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'HeatKernel';
W = lapgraph(Xt',manifold);
D = full(diag(sum(W,2)));
L = D-W;
L = [zeros(ms),zeros(mt); zeros(ms),Xt*L*Xt'];

% Initialize Q: parameter transfer and regularization |B-A|_F+|B|_F
Q = [eye(ms),-eye(mt);-eye(ms),2*eye(mt)];

% Initialize S: target components perservation
Ht = eye(nt)-1/(nt)*ones(nt,nt);
S = [zeros(ms),zeros(mt); zeros(ms),Xt*Ht*Xt'];

Ns=1/ns*onehot(Ys,unique(Ys)); Nt=zeros(nt,C);
if ~isempty(YtPseudo); Nt=1/nt*onehot(YtPseudo,unique(Ys)); end
Rmin=[Ns*Ns',-Ns*Nt';-Nt*Ns',Nt*Nt'];
Rmin = Rmin / norm(Rmin, 'fro');

Ms=[]; Mt=[];
for i=1:C
    Ms=[Ms,repmat(Ns(:,i),1,C-1)];
    idx=1:C; idx(i)=[];
    Mt=[Mt,Nt(:,idx)];
end
Rmax=[Ms*Ms',-Ms*Mt';-Mt*Ms',Mt*Mt'];
Rmax = Rmax / norm(Rmax,'fro');

X = [Xs,zeros(size(Xt));zeros(size(Xs)),Xt];
R = X*(Rmin-mu*Rmax)*X';



% Generalized eigendecompostion
Emin = alpha*P +  beta*L + rho*Q + R; % alpha*P + beta*L + rho*Q + R;
Emax = S + alpha*P0;
[W,~] = eigs(Emin+10^(-3)*eye(ms+mt), Emax, d, 'SM'); % SM: smallestabs

% Smallest magnitudes
A = W(1:ms, :);
B = W(ms+1:end, :);

% Embeddings
Zs = A'*Xs;
Zt = B'*Xt;
Zs=Zs';
Zt=Zt';

end

function y_onehot=onehot(y,class)
% Encode label to onehot form
% Input:
% y: label vector, N*1
% Output:
% y_onehot: onehot label matrix, N*C
nc=length(class);
y_onehot=zeros(length(y), nc);
for i=1:length(y)
    y_onehot(i, class==y(i))=1;
end
end

function K = kernel(ker,X,X2,gamma)

switch ker
    case 'linear'
        
        if isempty(X2)
            K = X'*X;
        else
            K = X'*X2;
        end
        
    case 'rbf'
        
        n1sq = sum(X.^2,1);
        n1 = size(X,2);
        
        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*(X'*X);
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-gamma*D);
        
    case 'sam'
        
        if isempty(X2)
            D = X'*X;
        else
            D = X'*X2;
        end
        K = exp(-gamma*acos(D).^2);
        
    otherwise
        error(['Unsupported kernel ' ker])
end
end