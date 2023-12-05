function[av_beta, av_alpha] = compute_GFE_analytical_cov(grp,Y,X,beta,alpha)
    [N,T,K] = size(X);
    Matgrp = dummyvar(grp);
    [N, G] = size(Matgrp);
    for t=1:T
        Xbar(:,t,:) = (Matgrp'*Matgrp)\Matgrp'*reshape(X(:,t,:),N,K);
        res_X(:,t,:) = reshape(X(:,t,:),N,K)-Matgrp*reshape(Xbar(:,t,:),G,K);
    end
    residuals = Y-Matgrp*alpha;
    for k=1:K
        residuals = residuals-X(:,:,k).*beta(k);
    end
    residuals = reshape(residuals, N, T);
    Sigma = zeros([K K]);
    Omega = zeros([K K]);
    for i=1:N
        for t=1:T
            Sigma = Sigma + reshape(res_X(i,t,:),[K,1])*reshape(res_X(i,t,:),[K,1])'./ (N*T);
            for s=1:T
                Omega = Omega + residuals(i,t).*residuals(i,s).*reshape(res_X(i,t,:),[K,1])*reshape(res_X(i,s,:),[K,1])'./ (N*T);
            end
        end
    end
    Sigma_inv = inv(Sigma);
    av_beta = Sigma_inv*Omega*Sigma_inv./(N*T);
    for t=1:T
        av_alpha(:,t) = (Matgrp'*residuals(:,t)).^2 ./ (sum(Matgrp,[1]).^2)'./N;
    end
end
