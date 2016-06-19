% demo nonparametric clustering with DPM using Large Sampling Asymptotic
% author: Dr Vu Nguyen, Deakin University, prada-research.net/~tienvu
% email: v.nguyen@deakin.edu.au

clear all;
close all;

disp('==============================================================');
disp('DPM Large Sample Asymptotic for Nonparametric Clustering');

load('synthetic_lsa.mat');

[NN,DD]=size(Feature);

fprintf('Clustering %d data points, each contains %d features. We will estimate the unknown number of clusters.\n',NN,DD);

% shuffle the data and groundtruth label
idx=randperm(NN);
Feature=Feature(idx,:);
Label=Label(idx);




%%
disp('===============================================================');
disp('Start estimating lambda');
tic

% this initialization of initK to find the suitable lambda value
initK=9;

% append the epsilon value to zero entries in the feature to prevent
% infinity in computing KL divergence
eps=1e-4;
Feature(Feature==0)=eps;
Feature = spdiags(1./sum(Feature,2),0,NN,NN)*Feature;

clear eps, clear idx;

%T=mean(Feature);
T=zeros(initK+1,DD);
T(1,:)=mean(Feature);
Tindex=zeros(1,initK);

% select 10000 data points to estimate lambda
if NN>2000
    selectedIdx=linspace(1,NN-1,2000);
    selectedIdx=ceil(selectedIdx);
else
    selectedIdx=[1:NN];
end
   
for kk=1:initK

    maxIndex=0;
    maxDist=0;

    for j=1:length(selectedIdx)
        if ~ismember(selectedIdx(j),Tindex)
            fmat=ones(length(Tindex)+1,1)*Feature(selectedIdx(j),:);
            mydist=mat_kl_div_vu(fmat,T);
            [minValue ]=min(mydist);
            if minValue>maxDist
                maxDist=minValue;
                maxIndex=selectedIdx(j);
            end
        end
    end
    Tindex(kk)=maxIndex;
    T(kk+1,:)=Feature(maxIndex,:);
end
lambda=1*maxDist;


fprintf('The estimated lambda = %.2f\n',lambda);
timeLambda=toc;

clear j,clear kk, clear selectedIdx, clear temp,clear fmat;
%% run the main algorithm

% zz is the estimated clustering label: [NN x 1]
% KK is the number of estimated clusters: length(unique(zz))
% topic is the estimated topics: [KK x DD] where DD is the dimension

tic;
disp('==============================================================');
disp('Start running LSA');
[zz, KK, topics]=DPM_LargeSampleAsymptotic(Feature,lambda,'verbose',1);
timeLSA=toc;
disp('Finish running LSA');

%% evaluate the clustering score
fprintf('DPM LSA, #Cluster=%d\n',KK);
fprintf('DPM LSA Running Time [Estimating Lambda=%.1f sec] [Running LSA=%.1f sec]\n',timeLambda,timeLSA);

[purity, NMI, RI, Fscore, ARI] = cluster_evaluate_vectorized(zz,Label);
fprintf('Clustering Evaluation Purity=%.2f, NMI=%.2f, RI=%.2f, Fscore=%.2f, ARI=%.2f\n',purity, NMI, RI, Fscore, ARI);
%histc(zz,1:KK)



%% visualize the pattern
figure;
for kk=1:KK
    subplot(2,ceil(KK/2),kk);
    imagesc(reshape(topics(kk,:),5,5));
end
