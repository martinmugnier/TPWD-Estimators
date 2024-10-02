function [G,g] = TPWD_clustering(Y,c,link,normalized)
     % TPWD_CLUSTERING Return the triad pairwise-differencing (TPWD) 
     % clustering estimator proposed in Mugnier (2022).
     %
     % Author: Martin Mugnier
     % Email: martin.mugnier (at) psemail (dot) eu
     % (last update: August 2024)
     %
     % INPUTS: 
     % --------
     % Y          : NxT array of possibly unbalanced panel data outcomes; 
     % c          : nonnegative scalar thresholding parameter;  
     % link       : 'single', 'complete', 'average';
     % normalized : 0 (raw distance) or 1 (self-normalized distance).
     %
     % OUTPUTS:
     % --------
     % G          : scalar estimated number of groups;
     % g          : Nx1 array of estimated group memberships.
     % 
     % REFERENCE:
     % ----------
     % Mugnier, M. (2022), A Simple and Computationally Trivial Estimator 
     % for Grouped Fixed Effects Models.
     
     if nargin<4
        normalized = 0; %default choice is raw distance
     end 
     
     if nargin<3
        link = 'average'; %default linkage is 'average' 
     end
     
     [N,T] = size(Y);
     if sum(isnan(Y),'all')>0
         warning(['Missing values detected. A small value of ' ...
             'c may result in a lack of identification.']);
     end
     if any(sum(isnan(Y),2)>T-2)
         warning('Some units have less than two observations.');
     end
     % compute matrix of pairwise outcome differences at each time period
     s = permute(Y-permute(Y,[3 2 1]),[1 3 2]);
     if normalized
         s = s./std(s,0,3);
     end
     % compute dissimilarity matrix using pairwise differences and triads 
     D = abs(mean(reshape(Y,N,1,1,T).*reshape(s,1,N,N,T),4,...
         'omitnan'));
     for j=1:N
         D(j,j,:) = 0;
         D(j,:,j) = 0;
     end
     D = squareform(squeeze(max(D,[],1)));
     % perform agglomerative clustering
     W = linkage(D,link);
     g = cluster(W,'cutoff',c,'Criterion','distance');
     G = length(unique(g));
end