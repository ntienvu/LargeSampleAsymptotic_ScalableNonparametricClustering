function [purity, NMI, RI, Fscore, ARI] = cluster_evaluate_vectorized(clusterID, groundtruthID)
% Clustering Evaluation
% Input: 
% clusterID is the estimated clusters: [N x 1]
% groundtruthID is the groundtruth cluster: [N x 1]
% Output
% compute 5 clustering metrics: purity, Normalized Mutual Information (NMI)
% Rand Index (RI), Fscore and % Adjusted Rand Index (ARI)

TP = 0;
TN = 0;
FP = 0;
FN = 0;
N = numel(clusterID);
if(numel(clusterID) ~= numel(groundtruthID))
    purity = 0;
    NMI = 0;
    RI = 0;
    Fscore = 0;
    ARI = 0;
    disp('Warning:: The cluster and groundtruth labels have different length');
    return;
end

cluster_list = unique(clusterID);
gt_list = unique(groundtruthID);

[~, clusterID] = ismember(clusterID, cluster_list);
[~, groundtruthID] = ismember(groundtruthID, gt_list);
cluster_num = length(cluster_list);

sub_gt = cell(cluster_num,1);
sub_gt_list = cell(cluster_num,1);
hist_sub_gt = cell(cluster_num,1);

clusterNum = length(unique(clusterID));
clusterLen = zeros(1, clusterNum);
groundtruthNum = length(unique(groundtruthID));
groundtruthLen = zeros(groundtruthNum,1);

for i = 1:groundtruthNum
    groundtruthLen(i) = sum(groundtruthID==i);
end

MI = 0;
purity = 0;
for i = 1:cluster_num
    idx = clusterID==i;
    clusterLen(i) = sum(idx);
    sub_gt{i} = groundtruthID(idx);
    sub_gt_list{i} = unique(sub_gt{i});
    hist_sub_gt{i} = histc(sub_gt{i}, sub_gt_list{i});
    if(size(hist_sub_gt{i},1) < size(hist_sub_gt{i},2))
        hist_sub_gt{i} = hist_sub_gt{i}';
    end
    TP = TP + sum(hist_sub_gt{i} .* (hist_sub_gt{i}-1) / 2);
    FP = FP + (sum(sum(hist_sub_gt{i}*hist_sub_gt{i}')) - hist_sub_gt{i}'*hist_sub_gt{i})/2;
    purity = purity + max(hist_sub_gt{i});
    MI = MI + sum(hist_sub_gt{i} / N .* (log(N*hist_sub_gt{i}./(clusterLen(i)*groundtruthLen(sub_gt_list{i})))));
end

for i = 1:cluster_num
    for j = i+1:cluster_num
        for k = 1:length(sub_gt_list{i})
            for m = 1:length(sub_gt_list{j})
                if(sub_gt_list{i}(k) == sub_gt_list{j}(m))
                    FN = FN + hist_sub_gt{i}(k) * hist_sub_gt{j}(m);
                else
                    TN = TN + hist_sub_gt{i}(k) * hist_sub_gt{j}(m);
                end
            end
        end
    end
end

precision = TP / (TP+FP);
recall = TP / (TP+FN);

Fscore = 2 * precision * recall / (precision + recall);

RI = (TP + TN)/(TP + FP + TN + FN);
purity = purity / N;
% evaluate purity and NMI


entropy_cluster = 0;
entropy_groundtruth = 0;

for i = 1:clusterNum
    entropy_cluster = entropy_cluster - clusterLen(i) / N * log(clusterLen(i) / N);
end

for i = 1:groundtruthNum
    entropy_groundtruth = entropy_groundtruth - groundtruthLen(i) / N * log(groundtruthLen(i) / N);
end


NMI = MI / ((entropy_cluster + entropy_groundtruth)/2);

[ARI, ~, ~, ~] = AdjustedRandIndex(clusterID, groundtruthID);
end

function [AR,RI,MI,HI]=AdjustedRandIndex(c1,c2)
%RANDINDEX - calculates Rand Indices to compare two partitions
% ARI=RANDINDEX(c1,c2), where c1,c2 are vectors listing the 
% class membership, returns the "Hubert & Arabie adjusted Rand index".
% [AR,RI,MI,HI]=RANDINDEX(c1,c2) returns the adjusted Rand index, 
% the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
%
% See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of 
% Classification 2:193-218

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

if nargin < 2 | min(size(c1)) > 1 | min(size(c2)) > 1
   error('RandIndex: Requires two vector arguments')
   return
end

C=Contingency(c1,c2);	%form contingency matrix

n=sum(sum(C));
nis=sum(sum(C,2).^2);		%sum of squares of sums of rows
njs=sum(sum(C,1).^2);		%sum of squares of sums of columns

t1=nchoosek(n,2);		%total number of pairs of entities
t2=sum(sum(C.^2));	%sum over rows & columnns of nij^2
t3=.5*(nis+njs);

%Expected index (for adjustment)
nc=(n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1));

A=t1+t2-t3;		%no. agreements
D=  -t2+t3;		%no. disagreements

if t1==nc
   AR=0;			%avoid division by zero; if k=1, define Rand = 0
else
   AR=(A-nc)/(t1-nc);		%adjusted Rand - Hubert & Arabie 1985
end

RI=A/t1;			%Rand 1971		%Probability of agreement
MI=D/t1;			%Mirkin 1970	%p(disagreement)
HI=(A-D)/t1;	%Hubert 1977	%p(agree)-p(disagree)
end
function Cont=Contingency(Mem1,Mem2)

if nargin < 2 | min(size(Mem1)) > 1 | min(size(Mem2)) > 1
   error('Contingency: Requires two vector arguments')
   return;
end

Cont=zeros(max(Mem1),max(Mem2));

for i = 1:length(Mem1);
   Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
end
end