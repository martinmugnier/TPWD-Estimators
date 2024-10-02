function res = clustering_accuracy(grp_true,grp_hat)
    % CLUSTERING_ACCURACY Return the Precision rate, Recall rate, and Rand 
    % index of the estimated grouping grp_hat relative to the true
    % grouping grp_true.
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS: 
    % -------
    % grp_true   : Nx1 array of 'true' group memberships;
    % grp_hat    : Nx1 array of estimated group memberships.
    %
    % OUTPUT:
    % -------
    % res        : 1x3 array (Precision rate, Recall rate and Rand index).
    %
    % REFERENCE:
    % ----------
    % Mugnier (2022), A Simple and Computationally Trivial Estimator for 
    % Grouped Fixed Effects Models.
    
    dist_true = abs(bsxfun(@minus,grp_true,grp_true'))>0;
    dist_hat = abs(bsxfun(@minus,grp_hat,grp_hat'))>0;
    FP = sum((1-dist_hat).*dist_true,'all')/2;
    TP = (sum((1-dist_hat).*(1-dist_true),'all')-length(grp_true))/2;
    FN = sum(dist_hat.*(1-dist_true),'all')/2;
    TN = sum(dist_hat.*dist_true,'all')/2;
    if (TP+FP)>0
        P = TP/(TP+FP); %Precision
    else
        P = 0;
    end
    if (FN+TP)>0
        R = TP/(FN+TP); %Recall
    else
        R = 0;
    end
    RI = (TP+TN)/(TP+FP+FN+TN); %RandIndex
    res = [P R RI];
end