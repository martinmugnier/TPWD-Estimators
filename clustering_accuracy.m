function y = clustering_accuracy(grp_true, grp_hat)
    dist_true = abs(bsxfun(@minus,grp_true,grp_true'))>0;
    dist_hat = abs(bsxfun(@minus,grp_hat,grp_hat'))>0;
    FP = sum((1-dist_hat).*dist_true,'all')/2;
    TP = (sum((1-dist_hat).*(1-dist_true),'all')-length(grp_true))/2;
    FN = sum(dist_hat.*(1-dist_true),'all')/2;
    TN = sum(dist_hat.*dist_true,'all')/2;
    P = TP/(TP+FP); %Precision
    R = TP/(FN+TP); %Recall
    RI = (TP+TN)/(TP+FP+FN+TN); %RandIndex
    y = [P R RI];
end