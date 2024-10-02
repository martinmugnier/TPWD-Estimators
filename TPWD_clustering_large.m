function [G,g] = TPWD_clustering_large(Y,c,link,normalized,parallel)
     % TPWD_CLUSTERING_LARGE Return the triad pairwise-differencing (TPWD) 
     % clustering estimator proposed in Mugnier (2022). To save memory and 
     % time for applications to large datasets, implementation is 
     % not fully vectorized.
     %
     % Author: Martin Mugnier
     % Email: martin.mugnier (at) psemail (dot) eu
     % (last update: August 2024)
     %
     % INPUTS:
     % -------
     % Y          : NxT array of possibly unbalanced panel data outcomes; 
     % c          : nonnegative scalar thresholding parameter;  
     % link       : 'single', 'complete', 'average';
     % normalized : 0 (raw distance)  or 1 (self-normalized distance).
     % parallel   : run in parallel if true.
     %
     % OUTPUTS:
     % --------
     % G          : scalar estimated number of groups;
     % g          : Nx1 array of estimated group memberships.
     % REFERENCE:
     % ----------
     % Mugnier, M. (2022), A Simple and Computationally Trivial Estimator 
     % for Grouped Fixed Effects Models.
     
     if nargin<5
        parallel = false; %default choice is not in parallel 
     end 
    
     if nargin<4
        normalized = 0; %default choice is not normalized  
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
     % compute dissimilarity matrix
     D = zeros(N,N);
     pairs_unique = nchoosek(1:N,2);
     D_tilde = zeros(length(pairs_unique),1);
     if normalized
         if parallel
             parfor(j=1:length(pairs_unique),20)
                 idx1 = pairs_unique(j,1);
                 idx2 = pairs_unique(j,2);
                 % compute matrix of pairwise outcome differences at each time 
                 diff_vec = bsxfun(@times, Y(setdiff(1:N,...
                     [idx1 idx2]),:), Y(idx1,:)-Y(idx2,:));
                 diff_vec = diff_vec./std(diff_vec,0,2,'omitnan');
                 pairwise_comp = abs(mean(diff_vec,2,'omitnan'));
                 tpwd_dist = squeeze(max(pairwise_comp,[],1));
                 D_tilde(j) = tpwd_dist;
             end
             for j=1:length(pairs_unique)
                 idx1 = pairs_unique(j,1);
                 idx2 = pairs_unique(j,2);
                 D(idx1,idx2) = D_tilde(j);
                 D(idx2,idx1) = D_tilde(j);
             end
         else
             for j=1:length(pairs_unique)
                 idx1 = pairs_unique(j,1);
                 idx2 = pairs_unique(j,2);
                 % compute matrix of pairwise outcome differences at each time 
                 diff_vec = bsxfun(@times, Y(setdiff(1:N,...
                     [idx1 idx2]),:), Y(idx1,:)-Y(idx2,:));
                 diff_vec = diff_vec./std(diff_vec,0,2,'omitnan');
                 pairwise_comp = abs(mean(diff_vec,2,'omitnan'));
                 tpwd_dist = squeeze(max(pairwise_comp,[],1));
                 D(idx1,idx2) = tpwd_dist;
                 D(idx2,idx1) = tpwd_dist;
             end
         end
     else
         if parallel
             parfor(j=1:length(pairs_unique),20)
                 idx1 = pairs_unique(j,1);
                 idx2 = pairs_unique(j,2);
                 % compute matrix of pairwise outcome differences at each time 
                 diff_vec = bsxfun(@times, Y(setdiff(1:N,...
                     [idx1 idx2]),:), Y(idx1,:)-Y(idx2,:));
                 pairwise_comp = abs(mean(diff_vec,2,'omitnan'));
                 tpwd_dist = squeeze(max(pairwise_comp,[],1));
                 D_tilde(j) = tpwd_dist;
             end
             for j=1:length(pairs_unique)
                 idx1 = pairs_unique(j,1);
                 idx2 = pairs_unique(j,2);
                 D(idx1,idx2) = D_tilde(j);
                 D(idx2,idx1) = D_tilde(j);
             end
         else
             for j=1:length(pairs_unique)
                 idx1 = pairs_unique(j,1);
                 idx2 = pairs_unique(j,2);
                 % compute matrix of pairwise outcome differences at each time 
                 diff_vec = bsxfun(@times, Y(setdiff(1:N,...
                     [idx1 idx2]),:), Y(idx1,:)-Y(idx2,:));
                 pairwise_comp = abs(mean(diff_vec,2,'omitnan'));
                 tpwd_dist = squeeze(max(pairwise_comp,[],1));
                 D(idx1,idx2) = tpwd_dist;
                 D(idx2,idx1) = tpwd_dist;
             end
         end
     end
     D = squareform(D);
     % perform agglomerative clustering
     W = linkage(D,link);
     g = cluster(W,'cutoff',c,'Criterion','distance');
     G = length(unique(g));
end