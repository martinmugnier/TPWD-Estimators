%--------------------- MONTE CARLO SIMULATIONS ----------------------------
% This code produces Tables:
%  - S9,S10: iid errors, signal-to-noise ratio=1, lagged outcomes.
%--------------------------------------------------------------------------

clear;
rng('default'); 

G = 3; 
%G = 4;  %Unmute if G=4
beta = [1 0.1]; 
K = 2;
Tseq = [7,10,20,40]; 
sigma = 1/3;

%Simulations
nsim = 500;
res = [];

G_hat_res = [];
grp_hat_res = [];
betahat_res = [];
alphahat_res = [];
betahat_oracle_res = [];
alphahat_oracle_res = [];
beta_nnr_res = [];

%Panel A 
N = 90;
grp = repelem([1:G],N/G);
%grp = 1+((1:N)>22)+((1:N)>44)+((1:N)>66); %Unmute if G=4
grp_dum_mat = dummyvar(grp);
for idxt=1:4
    alpha = [];
    T = Tseq(idxt);
    alpha(1,:) = ones(1,T); 
    alpha(2,:) = ((1:T)-1)/(T-1);
    alpha(3,:) = zeros(1,T);
    split = floor(T/2);
    %alpha(4,:) = [zeros(1,split-1) ((split:T)-split)/(T-split)]; %Unmute if G=4
    for simu=1:nsim
        disp([N,idxt,simu]);
        err = normrnd(0,sigma,N,T);
        %Initial conditions
        grp_initialmeans = [1 -1/T 0 0]'; 
        Y0 = normrnd(grp_dum_mat*grp_initialmeans(1:G),sigma/2); 
        X = normrnd(0,1/3,N,T,1)+0.2*grp_dum_mat*alpha;
        lag = Y0;
        Y = [];
        for t=1:T
            X(:,t,2) = lag;
            Y(:,t) = beta(1)*X(:,t,1)+beta(2)*X(:,t,2)+grp_dum_mat*alpha(:,t)+err(:,t);
            lag = Y(:,t);
        end 
        %Preliminary slope estimation
        psi = log(log(T))/(4*sqrt(min(N,T)));
        beta_nnr = nucnorm_reg(Y,X,psi);
        beta_nnr_res(simu,idxt,:) = beta_nnr;
        res(simu,idxt,1) = beta_nnr(1)-beta(1);
        res(simu,idxt,2) = beta_nnr(2)-beta(2);
        res(simu,idxt,3) = (beta_nnr(1)-beta(1))^2;
        res(simu,idxt,4) = (beta_nnr(2)-beta(2))^2;
        %Clustering
        residual = Y-sum(X.*reshape(kron(beta_nnr',ones([N,T])),N,T,K),3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat1,grphat1,~] = tpwd_pureGFE(residual,c_opt,2);
        %Pooled OLS
        [betahat1,~] =  FE_reg_withcov(Y,X,grphat1,true);
        %Iterate 2
        residual = Y-sum(X.*reshape(kron(betahat1',ones([N,T])),N,T,K), 3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat2,grphat2,~] = tpwd_pureGFE(residual,c_opt,2);
        [betahat2,~] =  FE_reg_withcov(Y,X,grphat2,true);
        %Iterate 3
        residual = Y-sum(X.*reshape(kron(betahat2',ones([N,T])),N,T,K), 3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat3,grphat3,~] = tpwd_pureGFE(residual,c_opt,2);
        [betahat3,~] =  FE_reg_withcov(Y,X,grphat3,true);
        %Iterate 4
        residual = Y-sum(X.*reshape(kron(betahat3',ones([N,T])),N,T,K), 3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat4,grphat4,~] = tpwd_pureGFE(residual,c_opt,2);
        [betahat4,alphahat4] =  FE_reg_withcov(Y,X,grphat4,true);
        G_hat_res(simu,idxt,1) = Ghat4;
        grp_hat_res(simu,idxt,1:N) = grphat4;
        betahat_res(simu,idxt,:) = betahat4;
        alphahat_res(simu,idxt,1:Ghat4,1:T) = alphahat4;
        res(simu,idxt,5) = Ghat4;
        res(simu,idxt,6) = betahat4(1)-beta(1);
        res(simu,idxt,7) = betahat4(2)-beta(2);
        res(simu,idxt,8) = (betahat4(1)-beta(1))^2;
        res(simu,idxt,9) = (betahat4(2)-beta(2))^2;
        res(simu,idxt,10) = sqrt(mean((grp_dum_mat*alpha- ...
                    dummyvar(grphat4)*alphahat4).^2,'all')); 
        [A,~] = compute_GFE_analytical_cov(grphat4,Y,X,betahat4,alphahat4);
        se_tpwd1 = sqrt(A(1,1));
        se_tpwd2 = sqrt(A(2,2));
        res(simu,idxt,11) = (beta(1)>=betahat4(1)-se_tpwd1*norminv(1-0.05/2))...
            *(beta(1)<=betahat4(1)+se_tpwd1*norminv(1-0.05/2)); 
        res(simu,idxt,12) = (beta(2)>=betahat4(2)-se_tpwd2*norminv(1-0.05/2))...
            *(beta(2)<=betahat4(2)+se_tpwd2*norminv(1-0.05/2));
        [betahat_oracle,alphahat_oracle] =  FE_reg_withcov(Y,X,grp',true);
        beta_hat_oracle_res(simu,idxt,:) = betahat_oracle;
        alphahat_oracle_res(simu,idxt,1:G,1:T) = alphahat_oracle;
        res(simu,idxt,13) = betahat_oracle(1)-beta(1);
        res(simu,idxt,14) = betahat_oracle(2)-beta(2);
        res(simu,idxt,15) = (betahat_oracle(1)-beta(1))^2;
        res(simu,idxt,16) = (betahat_oracle(2)-beta(2))^2;
        res(simu,idxt,17) = sqrt(mean((grp_dum_mat*alpha- ...
            grp_dum_mat*alphahat_oracle).^2,'all'));
        [A,~] = compute_GFE_analytical_cov(grp,Y,X,betahat_oracle,alphahat_oracle);
        se_oracle1 = sqrt(A(1,1));
        se_oracle2 = sqrt(A(2,2));
        res(simu,idxt,18) = (beta(1)>=betahat_oracle(1)-se_oracle1...
            *norminv(1-0.05/2))*(beta(1)<=betahat_oracle(1)+se_oracle1*norminv(1-0.05/2)); 
        res(simu,idxt,19) = (beta(2)>=betahat_oracle(2)-se_oracle2...
            *norminv(1-0.05/2))*(beta(2)<=betahat_oracle(2)+se_oracle2*norminv(1-0.05/2));
        res(simu,idxt,20:22) = clustering_accuracy(grp,grphat4); 
    end
end

%Panel B 
N = 180;
grp = repelem([1:G],N/G);
%grp = 1+((1:N)>22)+((1:N)>44)+((1:N)>66); %Unmute if G=4
grp_dum_mat = dummyvar(grp);
for idxt=5:8
    alpha = [];
    T = Tseq(idxt-4);
    alpha(1,:) = ones(1,T); 
    alpha(2,:) = ((1:T)-1)/(T-1);
    alpha(3,:) = zeros(1,T);
    split = floor(T/2);
    %alpha(4,:) = [zeros(1,split-1) ((split:T)-split)/(T-split)]; %Unmute if G=4
    for simu=1:nsim
        disp([N,idxt,simu]);
        err = normrnd(0,sigma,N,T);
        %Initial conditions
        grp_initialmeans = [1 -1/T 0 0]'; 
        Y0 = normrnd(grp_dum_mat*grp_initialmeans(1:G),sigma/2); 
        X = normrnd(0,1/3,N,T,1)+0.2*grp_dum_mat*alpha;
        lag = Y0;
        Y = [];
        for t=1:T
            X(:,t,2) = lag;
            Y(:,t) = beta(1)*X(:,t,1)+beta(2)*X(:,t,2)+grp_dum_mat*alpha(:,t)+err(:,t);
            lag = Y(:,t);
        end 
        %Preliminary slope estimation
        psi = log(log(T))/(4*sqrt(min(N,T)));
        beta_nnr = nucnorm_reg(Y,X,psi);
        beta_nnr_res(simu,idxt,:) = beta_nnr;
        res(simu,idxt,1) = beta_nnr(1)-beta(1);
        res(simu,idxt,2) = beta_nnr(2)-beta(2);
        res(simu,idxt,3) = (beta_nnr(1)-beta(1))^2;
        res(simu,idxt,4) = (beta_nnr(2)-beta(2))^2;
        %Clustering
        residual = Y-sum(X.*reshape(kron(beta_nnr',ones([N,T])),N,T,K),3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat1,grphat1,~] = tpwd_pureGFE(residual,c_opt,2);
        %Pooled OLS
        [betahat1,~] =  FE_reg_withcov(Y,X,grphat1,true);
        %Iterate 2
        residual = Y-sum(X.*reshape(kron(betahat1',ones([N,T])),N,T,K), 3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat2,grphat2,~] = tpwd_pureGFE(residual,c_opt,2);
        [betahat2,~] =  FE_reg_withcov(Y,X,grphat2,true);
        %Iterate 3
        residual = Y-sum(X.*reshape(kron(betahat2',ones([N,T])),N,T,K), 3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat3,grphat3,~] = tpwd_pureGFE(residual,c_opt,2);
        [betahat3,~] =  FE_reg_withcov(Y,X,grphat3,true);
        %Iterate 4
        residual = Y-sum(X.*reshape(kron(betahat3',ones([N,T])),N,T,K), 3);
        sigmahat = std(residual,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        [Ghat4,grphat4,~] = tpwd_pureGFE(residual,c_opt,2);
        [betahat4,alphahat4] =  FE_reg_withcov(Y,X,grphat4,true);
        G_hat_res(simu,idxt,1) = Ghat4;
        grp_hat_res(simu,idxt,1:N) = grphat4;
        betahat_res(simu,idxt,:) = betahat4;
        alphahat_res(simu,idxt,1:Ghat4,1:T) = alphahat4;
        res(simu,idxt,5) = Ghat4;
        res(simu,idxt,6) = betahat4(1)-beta(1);
        res(simu,idxt,7) = betahat4(2)-beta(2);
        res(simu,idxt,8) = (betahat4(1)-beta(1))^2;
        res(simu,idxt,9) = (betahat4(2)-beta(2))^2;
        res(simu,idxt,10) = sqrt(mean((grp_dum_mat*alpha- ...
                    dummyvar(grphat4)*alphahat4).^2,'all')); 
        [A,~] = compute_GFE_analytical_cov(grphat4,Y,X,betahat4,alphahat4);
        se_tpwd1 = sqrt(A(1,1));
        se_tpwd2 = sqrt(A(2,2));
        res(simu,idxt,11) = (beta(1)>=betahat4(1)-se_tpwd1*norminv(1-0.05/2))...
            *(beta(1)<=betahat4(1)+se_tpwd1*norminv(1-0.05/2)); 
        res(simu,idxt,12) = (beta(2)>=betahat4(2)-se_tpwd2*norminv(1-0.05/2))...
            *(beta(2)<=betahat4(2)+se_tpwd2*norminv(1-0.05/2));
        [betahat_oracle,alphahat_oracle] =  FE_reg_withcov(Y,X,grp',true);
        beta_hat_oracle_res(simu,idxt,:) = betahat_oracle;
        alphahat_oracle_res(simu,idxt,1:G,1:T) = alphahat_oracle;
        res(simu,idxt,13) = betahat_oracle(1)-beta(1);
        res(simu,idxt,14) = betahat_oracle(2)-beta(2);
        res(simu,idxt,15) = (betahat_oracle(1)-beta(1))^2;
        res(simu,idxt,16) = (betahat_oracle(2)-beta(2))^2;
        res(simu,idxt,17) = sqrt(mean((grp_dum_mat*alpha- ...
            grp_dum_mat*alphahat_oracle).^2,'all'));
        [A,~] = compute_GFE_analytical_cov(grp,Y,X,betahat_oracle,alphahat_oracle);
        se_oracle1 = sqrt(A(1,1));
        se_oracle2 = sqrt(A(2,2));
        res(simu,idxt,18) = (beta(1)>=betahat_oracle(1)-se_oracle1...
            *norminv(1-0.05/2))*(beta(1)<=betahat_oracle(1)+se_oracle1*norminv(1-0.05/2)); 
        res(simu,idxt,19) = (beta(2)>=betahat_oracle(2)-se_oracle2...
            *norminv(1-0.05/2))*(beta(2)<=betahat_oracle(2)+se_oracle2*norminv(1-0.05/2));
        res(simu,idxt,20:22) = clustering_accuracy(grp,grphat4); 
    end
end

save('mc_res_full_gfe_S9_S10_G3.mat','res')
save('mc_G_hat_res_full_gfe_S9_S10_G3.mat','G_hat_res')
save('mc_grp_hat_res_full_gfe_S9_S10_G3.mat','grp_hat_res')
save('mc_betahat_res_res_full_gfe_S9_S10_G3.mat','betahat_res')
save('mc_alphahat_res_full_gfe_S9_S10_G3.mat','alphahat_res')
save('mc_betahat_oracle_res_res_full_gfe_S9_S10_G3.mat','betahat_oracle_res')
save('mc_alphahat_oracle_res_full_gfe_S9_S10_G3.mat','alphahat_oracle_res')
save('mc_beta_nnr_res_full_gfe_S9_S10_G3.mat','beta_nnr_res')


% Table S9 (Full GFE: Ghat, Bias, RMSE, coverage)
tab1_out = reshape(mean(res(:,1:8,1:19)),8,19);
tab1_out(:,[3 4 8 9 15 16]) = sqrt(tab1_out(:,[3 4 8 9 15 16]));
disp(round(tab1_out,3));
save('TableS9_G3.mat','tab1_out')

% Table S10 (Full GFE: classification accuracy)
tab2_out = round(reshape(mean(res(:,1:8,20:22)),8,3),3);
disp(tab2_out);
save('TableS10_G3.mat','tab2_out')


