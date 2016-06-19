function [zz, KK, topics ]=DPM_LargeSampleAsymptotic(Feature,lambda,varargin)
%*****************************************************************
% Input
% Feature is the feature matrix: [NN x DD], should be nonnegative
% lambda is the penalization parameter: scalar
%' verbose' = 1 to print the intermediate result, 'verbose' = 0 otherwise
% Output
% zz is the estimated clustering label: [NN x 1]
% KK is the number of estimated clusters: length(unique(zz))
% topic is the estimated topics: [KK x DD] where DD is the dimension

% Dr Vu Nguyen, v.nguyen@deakin.edu.au
% reference: Large Sample Asymptotic for Nonparametric Mixture Model with Count Data. V Nguyen, D Phung, T Le, S Venkatesh 
% In Workshop on Advances in Approximate Bayesian Inference at Neural Information Processing Systems, (NIPS), 2015.
%*****************************************************************
warning off;
[verbose] = process_options(varargin,'verbose',0);

[NN,DD]=size(Feature);
eps=1e-6;
Feature(Feature==0)=eps;
Feature = spdiags(1./sum(Feature,2),0,NN,NN)*Feature;

% Init the values
KK=3;

%cluster indicators zi=1 for all i
zz=ceil(KK*rand(1,NN));

% topic: the global pattern (or topic)
topics=zeros(KK,DD);
for kk=1:KK
    %topics(kk,:)=mean(Feature(zz==kk,:))+regularizer;
    topics(kk,:)=mean(Feature(zz==kk,:));
    topics(kk,:)=topics(kk,:)./sum(topics(kk,:));
end

% the proportion of data points over clusters
N_k=histc(zz,1:KK);
%distance from xi to all clusters
%d=[];
convergence=false;

TT=30;
iter=1;

count_converged=1;

count_print=0;
while(~convergence && iter<TT)
    
    %fprintf(1, repmat('\b',1,count_print)); %delete line before
    count_print = fprintf('Iter=%d KK=%d Time: %s\n',iter,KK, datestr(now, 'HH:MM:SS'));
    
    iter=iter+1;
    convergence=true;
    
    % remove empty cluster
    for kk=KK:-1:1
        if N_k(kk)==0
            idx=find(zz>kk);
            zz(idx)=zz(idx)-1;
            topics(kk,:)=[];
            N_k(kk)=[];
            
            KK=KK-1;
            if verbose==1
                fprintf('removing cluster KK=%d\n',KK);
            end
        end
    end
    
    KK=length(N_k);
    for ii=1:NN
        
        xi=Feature(ii,:);
        
        % get the old label
        kk=zz(ii);
        % update the topic after removing a data point xi
        topics(kk,:) =(topics(kk,:)*N_k(kk)-xi)./(N_k(kk)-1);
        N_k(kk)=N_k(kk)-1;
        
        
        % remove the cluster if empty
        if N_k(kk)==0
            N_k(kk)=[];
            topics(kk,:)=[];
            
            idx=find(zz>kk);
            zz(idx)=zz(idx)-1;
            
            KK=KK-1;
            if verbose==1
                fprintf('removing cluster KK=%d\n',KK);
            end
        end
        
        % matrix operation to compute KL divergence from xi to all topics
        %a=ones(KK,1)*xi;
        dic=mat_kl_div_vu(xi,topics);
        
        % select the shortest distance
        [dic_min, dic_argmin]=min(dic);
        
        %if(dic_min>lambda*log(KK))
        if(dic_min>lambda)

            
            count_converged=0; % reset the counter
            %create a new cluster
            KK=KK+1;
            zz(ii)=KK;
            N_k(KK)=1;
            %topics=[topics; xi+regularizer]; %add the mean of new cluster
            topics=[topics; xi]; %add the mean of new cluster
            topics(end,:)=topics(end,:)./sum(topics(end,:));
            convergence=false;%going on...
            
            if verbose==1
                fprintf('adding new cluster KK=%d\n',KK);
            end
        else
            if(dic_argmin~=kk) % if the closet cluster of the data point is changed
                count_converged=0; % reset the counter
                % add current point to closest cluster
                
                % update the topic after adding xi
                zz(ii)=dic_argmin; % Put the point into cluster
                topics(dic_argmin,:) =(topics(dic_argmin,:)*N_k(dic_argmin)+xi)./(N_k(dic_argmin)+1); % Adjust the mean of cluster xi is newly put in
                
                N_k(zz(ii))=N_k(zz(ii))+1;
                
                convergence=false;
            else
                topics(kk,:) =(topics(kk,:)*N_k(kk)+xi)/(N_k(kk)+1); % update the topic after adding xi
                topics(kk,:)=topics(kk,:)./sum(topics(kk,:));
                N_k(kk)=N_k(kk)+1;
                
                count_converged=count_converged+1; % increment the counter
                if count_converged>ceil(0.1*NN) % check convergence
                    convergence=true;
                    fprintf('LSA converged -');
                    break;
                end
            end
        end
    end
    
    % merging two identical clusters as the effect of hard assignment
    % clustering
    if KK>2
        myDist=pdist(topics,@(Xi,Xj) mat_kl_div_vu(Xi,Xj));
        myDist=squareform(myDist);
        myDist=myDist+diag(ones(1,KK)*99);
        kk=1;
        while(kk<size(myDist,1))
            idx=find(myDist(kk,:)<0.005*DD);
            % merging kk to ll
            for jj=1:length(idx)
                ll=idx(jj);
                
                newTopic=( topics(kk,:)*N_k(kk)+topics(ll,:)*N_k(ll))./(N_k(kk)+N_k(ll));
                
                % assign cluster ll to kk
                zz(zz==ll)=kk;
                
                idx2 = find(zz>ll);
                zz(idx2) = zz(idx2) - 1;
                topics(ll,:)=[];
                
                N_k(kk)=N_k(kk)+N_k(ll);
                N_k(ll)=[];
                topics(kk,:)=newTopic;
                topics(kk,:)=topics(kk,:)./sum(topics(kk,:));
                myDist(ll,:)=[];
                myDist(:,ll)=[];
                KK=KK-1;
                if verbose==1
                    fprintf('merging cluster %d to cluster %d KK=%d\n',kk,ll,KK);
                end
                break;
            end
            kk=kk+1;
        end
    end
    
end

% again, remove empty cluster
for kk=KK:-1:1
    if N_k(kk)==0
        idx=find(zz>kk);
        zz(idx)=zz(idx)-1;
        topics(kk,:)=[];
        N_k(kk)=[];
    end
end

KK=length(N_k);
fprintf('final KK=%d.\n',KK);

end
