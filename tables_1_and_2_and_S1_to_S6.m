%--------------------- MONTE CARLO SIMULATIONS ----------------------------
% This code produces Tables:
%  - 1,2: iid errors, signal-to-noise ratio=1, average link;
%  - S1,S2: iid errors, signal-to-noise ratio=1, unbalanced groups;
%  - S3,S4: dependent errors, signal-to-noise ratio=1;
%  - S5,S6: iid errors, signal-to-noise ratio=1/2;
%
% Notes: as in Bonhomme and Manresa (2015) replication package, results
% can slightly vary due to how MATLAB handles random seeds in
% parallelized loops.
%
% Author: Martin Mugnier, Paris School of Economics
% Email: martin.mugnier (at) psemail (dot) eu 
% (last update: September 2024)
% 
% REFERENCE:
% ----------
% Mugnier, M. (2022), A Simple and Computationally Trivial Estimator for 
% Grouped Fixed Effects Models.
%--------------------------------------------------------------------------

clear;

rng('default');

%Settings
balanced_grp = 1; % Change to 0 to output Tables S1 and S2;
error_correlation = 0; % Change to 1 to output Tables S3 and S4;
sigma = 1/3; % Change to 2/3 to output Tables S5 and S6;

Gseq = [3,4];
Tseq = [7,10,20,40];

n_init_GFE = 500; % for Bonhomme and Manresa (2015) estimator
spectr_lbda_grid = linspace(1,50,100); % For Chetverikov and Manresa (2022)'s spectral clustering

link = 'average'; % For Mugnier (2022)'s TPWD clustering

% Path to results
path = append('tables_1_and_2/',link);
%path = append('tables_S1_and_S2/',link);
%path = append('tables_S3_and_S4/',link);
%path = append('tables_S5_and_S6/',link);

%Simulations
ncores = 30;
nsim = 500;

res_full_tab1 = nan(nsim,16,16);
res_full_tab2 = nan(nsim,16,27);

for G=Gseq
    res_Ghat_TPWD = zeros(nsim,8); % TPWD
    res_Ghat_S2 = zeros(nsim,8);   % Spectral clustering, g=2
    res_Ghat_S3 = zeros(nsim,8);   % Spectral clustering, g=3
    res_Ghat_S4 = zeros(nsim,8);   % Spectral clustering, g=4
    res_Ghat_S10 = zeros(nsim,8);  % Spectral clustering, g=10 (not feasible if T<10)

    res_CPU = zeros(nsim,8); % TPWD

    res_grp_hat1 = zeros(nsim,8,180);  % TPWD
    res_grp_hat2 = zeros(nsim,8,180);  % Spectral clustering, g=2 
    res_grp_hat3 = zeros(nsim,8,180);  % Spectral clustering, g=3 
    res_grp_hat4 = zeros(nsim,8,180);  % Spectral clustering, g=4 
    res_grp_hat5 = zeros(nsim,8,180);  % Spectral clustering, g=10 (not feasible if T<10) 
    res_grp_hat6 = zeros(nsim,8,180);  % GFE, g=2
    res_grp_hat7 = zeros(nsim,8,180);  % GFE, g=3
    res_grp_hat8 = zeros(nsim,8,180);  % GFE, g=4
    res_grp_hat9 = zeros(nsim,8,180);  % GFE, g=10
    res_grp_hat10 = zeros(nsim,8,180); % pooled OLS oracle

    res_alpha_hat1 = zeros(nsim,8,180,40);
    res_alpha_hat2 = zeros(nsim,8,180,40);
    res_alpha_hat3 = zeros(nsim,8,180,40);
    res_alpha_hat4 = zeros(nsim,8,180,40);
    res_alpha_hat5 = zeros(nsim,8,180,40);
    res_alpha_hat6 = zeros(nsim,8,180,40);
    res_alpha_hat7 = zeros(nsim,8,180,40);
    res_alpha_hat8 = zeros(nsim,8,180,40);
    res_alpha_hat9 = zeros(nsim,8,180,40);
    res_alpha_hat10 = zeros(nsim,8,180,40);

    res_table1 = nan(nsim,8,16); 
    res_table2 = nan(nsim,8,18);

    res_loop_alpha_hat1 = zeros(nsim,180,40);
    res_loop_alpha_hat2 = zeros(nsim,180,40);
    res_loop_alpha_hat3 = zeros(nsim,180,40);
    res_loop_alpha_hat4 = zeros(nsim,180,40);
    res_loop_alpha_hat5 = zeros(nsim,180,40);
    res_loop_alpha_hat6 = zeros(nsim,180,40);
    res_loop_alpha_hat7 = zeros(nsim,180,40);
    res_loop_alpha_hat8 = zeros(nsim,180,40);
    res_loop_alpha_hat9 = zeros(nsim,180,40);
    res_loop_alpha_hat10 = zeros(nsim,180,40);

    %Panel A
    N = 90;
    if G==3
        if balanced_grp==1
            grp = repelem(1:G,N/G)';
        else
            grp = (1+(1:N>2)+(1:N>N/10))';
        end
    else
        if balanced_grp==1
            grp = (1+(1:N>22)+(1:N>44)+(1:N>66))';
        else
           grp = (1+(1:N>2)+(1:N>N/10)+(1:N>N/2))';
        end
    end
    grp_dum_mat = dummyvar(grp);
    for idxt=1:4
        alpha = [];
        T = Tseq(idxt);
        alpha(1,:) = ones(1,T);
        alpha(2,:) = ((1:T)-1)/(T-1);
        alpha(3,:) = zeros(1,T);
        if G==4
            split = floor(T/2);
            alpha(4,:) = [zeros(1,split-1) ((split:T)-split)/(T-split)]; 
        end
        res_loop_grp_hat1 = zeros(nsim,N);
        res_loop_grp_hat2 = zeros(nsim,N);
        res_loop_grp_hat3 = zeros(nsim,N);
        res_loop_grp_hat4 = zeros(nsim,N);
        res_loop_grp_hat5 = zeros(nsim,N);
        res_loop_grp_hat6 = zeros(nsim,N);
        res_loop_grp_hat7 = zeros(nsim,N);
        res_loop_grp_hat8 = zeros(nsim,N);
        res_loop_grp_hat9 = zeros(nsim,N);
        res_loop_grp_hat10 = zeros(nsim,N);
        err = normrnd(0,sigma,N,T,nsim);
        if error_correlation==1
            for simu=1:nsim
                    noise_mod = arima('Constant',0,'AR',{0.20},'Variance',sigma^2); 
                    err(:,:,simu) = simulate(noise_mod,T,'NumPaths',N)'; 
            end
        end
        parfor (simu=1:nsim,ncores)
        %for simu=1:nsim
            disp([N,idxt,simu]);
            rng(2024+simu); % seed at the parallel worker level
            Y = grp_dum_mat*alpha+err(:,:,simu);
            sigma_hat = std(Y,0,'all'); 
            c_opt = 1.5*sigma_hat^2*log(T)/sqrt(T);
            tic
            [res_Ghat_TPWD(simu,idxt),alpha_hat1,res_loop_grp_hat1(simu,:)] = ...
                TPWD_estimator_without_covariates(Y,c_opt,link,false); 
            res_CPU(simu,idxt) = toc;
            [res_Ghat_S2(simu,idxt),res_loop_grp_hat2(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),2,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
            alpha_hat2 = FE_reg_nocov(Y,res_loop_grp_hat2(simu,:)');
            [res_Ghat_S3(simu,idxt),res_loop_grp_hat3(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),3,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
            alpha_hat3 = FE_reg_nocov(Y,res_loop_grp_hat3(simu,:)');
            [res_Ghat_S4(simu,idxt),res_loop_grp_hat4(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),4,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
            alpha_hat4 = FE_reg_nocov(Y,res_loop_grp_hat4(simu,:)');
            if min(N,T)>=10
                [res_Ghat_S10(simu,idxt),res_loop_grp_hat5(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),10,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
                alpha_hat5 = FE_reg_nocov(Y,res_loop_grp_hat5(simu,:)');
            end
            [alpha_hat6,res_loop_grp_hat6(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,2,n_init_GFE);
            [alpha_hat7,res_loop_grp_hat7(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,3,n_init_GFE);        
            [alpha_hat8,res_loop_grp_hat8(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,4,n_init_GFE);        
            [alpha_hat9,res_loop_grp_hat9(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,10,n_init_GFE);
            alpha_hat10 = FE_reg_nocov(Y,grp);
            res_loop_grp_hat10(simu,:) = grp;

            % handle arrays of changing size in parfor loop
            alpha_hat1_pad = zeros(180,40);
            alpha_hat2_pad = zeros(180,40);
            alpha_hat3_pad = zeros(180,40);
            alpha_hat4_pad = zeros(180,40);
            alpha_hat5_pad = zeros(180,40);
            alpha_hat6_pad = zeros(180,40);
            alpha_hat7_pad = zeros(180,40);
            alpha_hat8_pad = zeros(180,40);
            alpha_hat9_pad = zeros(180,40);
            alpha_hat10_pad = zeros(180,40);

            alpha_hat1_pad(1:res_Ghat_TPWD(simu,idxt),1:T) = alpha_hat1;
            alpha_hat2_pad(1:res_Ghat_S2(simu,idxt),1:T) = alpha_hat2;
            alpha_hat3_pad(1:res_Ghat_S3(simu,idxt),1:T) = alpha_hat3;
            alpha_hat4_pad(1:res_Ghat_S4(simu,idxt),1:T) = alpha_hat4;
            if min(N,T)>=10
                alpha_hat5_pad(1:res_Ghat_S10(simu,idxt),1:T) = alpha_hat5;
            end
            alpha_hat6_pad(1:2,1:T) = alpha_hat6;
            alpha_hat7_pad(1:3,1:T) = alpha_hat7;
            alpha_hat8_pad(1:4,1:T) = alpha_hat8;
            alpha_hat9_pad(1:10,1:T) = alpha_hat9;
            alpha_hat10_pad(1:G,1:T) = alpha_hat10;

            res_loop_alpha_hat1(simu,:,:) = alpha_hat1_pad;
            res_loop_alpha_hat2(simu,:,:) = alpha_hat2_pad;
            res_loop_alpha_hat3(simu,:,:) = alpha_hat3_pad;
            res_loop_alpha_hat4(simu,:,:) = alpha_hat4_pad;
            res_loop_alpha_hat5(simu,:,:) = alpha_hat5_pad;
            res_loop_alpha_hat6(simu,:,:) = alpha_hat6_pad;
            res_loop_alpha_hat7(simu,:,:) = alpha_hat7_pad;
            res_loop_alpha_hat8(simu,:,:) = alpha_hat8_pad;
            res_loop_alpha_hat9(simu,:,:) = alpha_hat9_pad;
            res_loop_alpha_hat10(simu,:,:) = alpha_hat10_pad;
        end
        % store results and fill in table
        for simu=1:nsim
            res_grp_hat1(simu,idxt,1:N) = res_loop_grp_hat1(simu,:);
            res_grp_hat2(simu,idxt,1:N) = res_loop_grp_hat2(simu,:);
            res_grp_hat3(simu,idxt,1:N) = res_loop_grp_hat3(simu,:);
            res_grp_hat4(simu,idxt,1:N) = res_loop_grp_hat4(simu,:);
            res_grp_hat5(simu,idxt,1:N) = res_loop_grp_hat5(simu,:);
            res_grp_hat6(simu,idxt,1:N) = res_loop_grp_hat6(simu,:);
            res_grp_hat7(simu,idxt,1:N) = res_loop_grp_hat7(simu,:);
            res_grp_hat8(simu,idxt,1:N) = res_loop_grp_hat8(simu,:);
            res_grp_hat9(simu,idxt,1:N) = res_loop_grp_hat9(simu,:);
            res_grp_hat10(simu,idxt,1:N) = res_loop_grp_hat10(simu,:);

            res_alpha_hat1(simu,idxt,:,:) = res_loop_alpha_hat1(simu,:,:);
            res_alpha_hat2(simu,idxt,:,:) = res_loop_alpha_hat2(simu,:,:);
            res_alpha_hat3(simu,idxt,:,:) = res_loop_alpha_hat3(simu,:,:);
            res_alpha_hat4(simu,idxt,:,:) = res_loop_alpha_hat4(simu,:,:);
            res_alpha_hat5(simu,idxt,:,:) = res_loop_alpha_hat5(simu,:,:);
            res_alpha_hat6(simu,idxt,:,:) = res_loop_alpha_hat6(simu,:,:);
            res_alpha_hat7(simu,idxt,:,:) = res_loop_alpha_hat7(simu,:,:);
            res_alpha_hat8(simu,idxt,:,:) = res_loop_alpha_hat8(simu,:,:);
            res_alpha_hat9(simu,idxt,:,:) = res_loop_alpha_hat9(simu,:,:);
            res_alpha_hat10(simu,idxt,:,:) = res_loop_alpha_hat10(simu,:,:);

            res_table1(simu,idxt,1) = res_Ghat_TPWD(simu,idxt);
            res_table1(simu,idxt,2) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat1(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat1(simu,idxt,1:res_Ghat_TPWD(simu,idxt),1:T),...
                res_Ghat_TPWD(simu,idxt),T)).^2,'all'));
            res_table1(simu,idxt,3) = res_CPU(simu,idxt);
            res_table1(simu,idxt,4) = res_Ghat_S2(simu,idxt);
            res_table1(simu,idxt,5) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat2(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat2(simu,idxt,1:res_Ghat_S2(simu,idxt),1:T),...
                res_Ghat_S2(simu,idxt),T)).^2,'all'));
            res_table1(simu,idxt,6) = res_Ghat_S3(simu,idxt);
            res_table1(simu,idxt,7) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat3(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat3(simu,idxt,1:res_Ghat_S3(simu,idxt),1:T),...
                res_Ghat_S3(simu,idxt),T)).^2,'all'));
            res_table1(simu,idxt,8) = res_Ghat_S4(simu,idxt);
            res_table1(simu,idxt,9) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat4(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat4(simu,idxt,1:res_Ghat_S4(simu,idxt),1:T),...
                res_Ghat_S4(simu,idxt),T)).^2,'all'));
            if min(N,T)>=10
                res_table1(simu,idxt,10) = res_Ghat_S10(simu,idxt);
                res_table1(simu,idxt,11) = sqrt(mean((grp_dum_mat*alpha- ...
                    dummyvar(reshape(res_grp_hat5(simu,idxt,1:N),1,N))*...
                    reshape(res_alpha_hat5(simu,idxt,1:res_Ghat_S10(simu,idxt),1:T),...
                    res_Ghat_S10(simu,idxt),T)).^2,'all'));
            end
            res_table1(simu,idxt,12) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat6(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat6(simu,idxt,1:2,1:T),2,T)).^2,'all'));
            res_table1(simu,idxt,13) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat7(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat7(simu,idxt,1:3,1:T),3,T)).^2,'all'));
            res_table1(simu,idxt,14) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat8(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat8(simu,idxt,1:4,1:T),4,T)).^2,'all'));
            res_table1(simu,idxt,15) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat9(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat9(simu,idxt,1:10,1:T),10,T)).^2,'all'));
            res_table1(simu,idxt,16) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat10(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat10(simu,idxt,1:G,1:T),G,T)).^2,'all'));
            
            res_table2(simu,idxt,1:3) = clustering_accuracy(grp,reshape(...
                res_grp_hat1(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,4:6) = clustering_accuracy(grp,reshape(...
                res_grp_hat2(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,7:9) = clustering_accuracy(grp,reshape(...
                res_grp_hat3(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,10:12) = clustering_accuracy(grp,reshape(...
                res_grp_hat4(simu,idxt,1:N),1,N));
            if min(N,T)>=10
                res_table2(simu,idxt,13:15) = clustering_accuracy(grp,reshape(...
                    res_grp_hat5(simu,idxt,1:N),1,N));
            end
            res_table2(simu,idxt,16:18) = clustering_accuracy(grp,reshape(...
                res_grp_hat6(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,19:21) = clustering_accuracy(grp,reshape(...
                res_grp_hat7(simu,idxt,1:N),1,N));        
            res_table2(simu,idxt,22:24) = clustering_accuracy(grp,reshape(...
                res_grp_hat8(simu,idxt,1:N),1,N));  
            
                res_table2(simu,idxt,25:27) = clustering_accuracy(grp,reshape(...
                    res_grp_hat9(simu,idxt,1:N),1,N));
        end       
    end

    %Panel B 
    N = 180;
    if G==3
        if balanced_grp==1
            grp = repelem(1:G,N/G)';
        else
            grp = (1+(1:N>2)+(1:N>N/10))';
        end
    else
        if balanced_grp==1
            grp = (1+(1:N>22)+(1:N>44)+(1:N>66))';
        else
           grp = (1+(1:N>2)+(1:N>N/10)+(1:N>N/2))';
        end
    end
    grp_dum_mat = dummyvar(grp);
    for idxt=5:8
        alpha = [];
        T = Tseq(idxt-4);
        alpha(1,:) = ones(1,T);
        alpha(2,:) = ((1:T)-1)/(T-1);
        alpha(3,:) = zeros(1,T);
        if G==4
            split = floor(T/2);
            alpha(4,:) = [zeros(1,split-1) ((split:T)-split)/(T-split)]; 
        end
        res_loop_grp_hat1 = zeros(nsim,N);
        res_loop_grp_hat2 = zeros(nsim,N);
        res_loop_grp_hat3 = zeros(nsim,N);
        res_loop_grp_hat4 = zeros(nsim,N);
        res_loop_grp_hat5 = zeros(nsim,N);
        res_loop_grp_hat6 = zeros(nsim,N);
        res_loop_grp_hat7 = zeros(nsim,N);
        res_loop_grp_hat8 = zeros(nsim,N);
        res_loop_grp_hat9 = zeros(nsim,N);
        res_loop_grp_hat10 = zeros(nsim,N);
        err = normrnd(0,sigma,N,T,nsim);
        if error_correlation==1
            for simu=1:nsim
                    noise_mod = arima('Constant',0,'AR',{0.20},'Variance',sigma^2); 
                    err(:,:,simu) = simulate(noise_mod,T,'NumPaths',N)'; 
            end
        end
        parfor (simu=1:nsim,ncores)
        %for simu=1:nsim
            disp([N,idxt,simu]);
            rng(2024+simu); 
            Y = grp_dum_mat*alpha+err(:,:,simu);
            sigma_hat = std(Y,0,'all'); 
            c_opt = 1.5*sigma_hat^2*log(T)/sqrt(T);
            if (T>=20)
                tic
                [res_Ghat_TPWD(simu,idxt),alpha_hat1,res_loop_grp_hat1(simu,:)] = ...
                    TPWD_estimator_without_covariates(Y,c_opt,link,false);
                res_CPU(simu,idxt) = toc;
            else
                tic
                [res_Ghat_TPWD(simu,idxt),alpha_hat1,res_loop_grp_hat1(simu,:)] = ...
                    TPWD_estimator_without_covariates(Y,c_opt,link,false); %TRUE
                res_CPU(simu,idxt) = toc;
            end
            [res_Ghat_S2(simu,idxt),res_loop_grp_hat2(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),2,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
            alpha_hat2 = FE_reg_nocov(Y,res_loop_grp_hat2(simu,:)');
            [res_Ghat_S3(simu,idxt),res_loop_grp_hat3(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),3,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
            alpha_hat3 = FE_reg_nocov(Y,res_loop_grp_hat3(simu,:)');
            [res_Ghat_S4(simu,idxt),res_loop_grp_hat4(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),4,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
            alpha_hat4 = FE_reg_nocov(Y,res_loop_grp_hat4(simu,:)');
            if min(N,T)>=10
                [res_Ghat_S10(simu,idxt),res_loop_grp_hat5(simu,:)] = ...
                CM_spectral_clustering(Y,zeros(N,T,1),10,randi([0 1],N,1),...
                zeros(1,1),zeros(1,1),spectr_lbda_grid);
                alpha_hat5 = FE_reg_nocov(Y,res_loop_grp_hat5(simu,:)');
            end
            [alpha_hat6,res_loop_grp_hat6(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,2,n_init_GFE);
            [alpha_hat7,res_loop_grp_hat7(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,3,n_init_GFE);        
            [alpha_hat8,res_loop_grp_hat8(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,4,n_init_GFE);        
            [alpha_hat9,res_loop_grp_hat9(simu,:)] = ...
                BM_algo1_multiple_init_without_covariates(Y,10,n_init_GFE);
            alpha_hat10 = FE_reg_nocov(Y,grp);
            res_loop_grp_hat10(simu,:) = grp;

            % handle arrays of changing size in parfor loop
            alpha_hat1_pad = zeros(180,40);
            alpha_hat2_pad = zeros(180,40);
            alpha_hat3_pad = zeros(180,40);
            alpha_hat4_pad = zeros(180,40);
            alpha_hat5_pad = zeros(180,40);
            alpha_hat6_pad = zeros(180,40);
            alpha_hat7_pad = zeros(180,40);
            alpha_hat8_pad = zeros(180,40);
            alpha_hat9_pad = zeros(180,40);
            alpha_hat10_pad = zeros(180,40);

            alpha_hat1_pad(1:res_Ghat_TPWD(simu,idxt),1:T) = alpha_hat1;
            alpha_hat2_pad(1:res_Ghat_S2(simu,idxt),1:T) = alpha_hat2;
            alpha_hat3_pad(1:res_Ghat_S3(simu,idxt),1:T) = alpha_hat3;
            alpha_hat4_pad(1:res_Ghat_S4(simu,idxt),1:T) = alpha_hat4;
            if min(N,T)>=10
                alpha_hat5_pad(1:res_Ghat_S10(simu,idxt),1:T) = alpha_hat5;
            end
            alpha_hat6_pad(1:2,1:T) = alpha_hat6;
            alpha_hat7_pad(1:3,1:T) = alpha_hat7;
            alpha_hat8_pad(1:4,1:T) = alpha_hat8;
            alpha_hat9_pad(1:10,1:T) = alpha_hat9;
            alpha_hat10_pad(1:G,1:T) = alpha_hat10;

            res_loop_alpha_hat1(simu,:,:) = alpha_hat1_pad;
            res_loop_alpha_hat2(simu,:,:) = alpha_hat2_pad;
            res_loop_alpha_hat3(simu,:,:) = alpha_hat3_pad;
            res_loop_alpha_hat4(simu,:,:) = alpha_hat4_pad;
            res_loop_alpha_hat5(simu,:,:) = alpha_hat5_pad;
            res_loop_alpha_hat6(simu,:,:) = alpha_hat6_pad;
            res_loop_alpha_hat7(simu,:,:) = alpha_hat7_pad;
            res_loop_alpha_hat8(simu,:,:) = alpha_hat8_pad;
            res_loop_alpha_hat9(simu,:,:) = alpha_hat9_pad;
            res_loop_alpha_hat10(simu,:,:) = alpha_hat10_pad;
        end
        % store results and fill in table
        for simu=1:nsim
            res_grp_hat1(simu,idxt,1:N) = res_loop_grp_hat1(simu,:);
            res_grp_hat2(simu,idxt,1:N) = res_loop_grp_hat2(simu,:);
            res_grp_hat3(simu,idxt,1:N) = res_loop_grp_hat3(simu,:);
            res_grp_hat4(simu,idxt,1:N) = res_loop_grp_hat4(simu,:);
            res_grp_hat5(simu,idxt,1:N) = res_loop_grp_hat5(simu,:);
            res_grp_hat6(simu,idxt,1:N) = res_loop_grp_hat6(simu,:);
            res_grp_hat7(simu,idxt,1:N) = res_loop_grp_hat7(simu,:);
            res_grp_hat8(simu,idxt,1:N) = res_loop_grp_hat8(simu,:);
            res_grp_hat9(simu,idxt,1:N) = res_loop_grp_hat9(simu,:);
            res_grp_hat10(simu,idxt,1:N) = res_loop_grp_hat10(simu,:);

            res_alpha_hat1(simu,idxt,:,:) = res_loop_alpha_hat1(simu,:,:);
            res_alpha_hat2(simu,idxt,:,:) = res_loop_alpha_hat2(simu,:,:);
            res_alpha_hat3(simu,idxt,:,:) = res_loop_alpha_hat3(simu,:,:);
            res_alpha_hat4(simu,idxt,:,:) = res_loop_alpha_hat4(simu,:,:);
            res_alpha_hat5(simu,idxt,:,:) = res_loop_alpha_hat5(simu,:,:);
            res_alpha_hat6(simu,idxt,:,:) = res_loop_alpha_hat6(simu,:,:);
            res_alpha_hat7(simu,idxt,:,:) = res_loop_alpha_hat7(simu,:,:);
            res_alpha_hat8(simu,idxt,:,:) = res_loop_alpha_hat8(simu,:,:);
            res_alpha_hat9(simu,idxt,:,:) = res_loop_alpha_hat9(simu,:,:);
            res_alpha_hat10(simu,idxt,:,:) = res_loop_alpha_hat10(simu,:,:);

            res_table1(simu,idxt,1) = res_Ghat_TPWD(simu,idxt);
            res_table1(simu,idxt,2) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat1(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat1(simu,idxt,1:res_Ghat_TPWD(simu,idxt),1:T),...
                res_Ghat_TPWD(simu,idxt),T)).^2,'all'));
            res_table1(simu,idxt,3) = res_CPU(simu,idxt);
            res_table1(simu,idxt,4) = res_Ghat_S2(simu,idxt);
            res_table1(simu,idxt,5) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat2(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat2(simu,idxt,1:res_Ghat_S2(simu,idxt),1:T),...
                res_Ghat_S2(simu,idxt),T)).^2,'all'));
            res_table1(simu,idxt,6) = res_Ghat_S3(simu,idxt);
            res_table1(simu,idxt,7) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat3(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat3(simu,idxt,1:res_Ghat_S3(simu,idxt),1:T),...
                res_Ghat_S3(simu,idxt),T)).^2,'all'));
            res_table1(simu,idxt,8) = res_Ghat_S4(simu,idxt);
            res_table1(simu,idxt,9) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat4(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat4(simu,idxt,1:res_Ghat_S4(simu,idxt),1:T),...
                res_Ghat_S4(simu,idxt),T)).^2,'all'));
            if min(N,T)>=10
                res_table1(simu,idxt,10) = res_Ghat_S10(simu,idxt);
                res_table1(simu,idxt,11) = sqrt(mean((grp_dum_mat*alpha- ...
                    dummyvar(reshape(res_grp_hat5(simu,idxt,1:N),1,N))*...
                    reshape(res_alpha_hat5(simu,idxt,1:res_Ghat_S10(simu,idxt),1:T),...
                    res_Ghat_S10(simu,idxt),T)).^2,'all'));
            end
            res_table1(simu,idxt,12) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat6(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat6(simu,idxt,1:2,1:T),2,T)).^2,'all'));
            res_table1(simu,idxt,13) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat7(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat7(simu,idxt,1:3,1:T),3,T)).^2,'all'));
            res_table1(simu,idxt,14) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat8(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat8(simu,idxt,1:4,1:T),4,T)).^2,'all'));
            res_table1(simu,idxt,15) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat9(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat9(simu,idxt,1:10,1:T),10,T)).^2,'all'));
            res_table1(simu,idxt,16) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat10(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat10(simu,idxt,1:G,1:T),G,T)).^2,'all'));
            
            res_table2(simu,idxt,1:3) = clustering_accuracy(grp,reshape(...
                res_grp_hat1(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,4:6) = clustering_accuracy(grp,reshape(...
                res_grp_hat2(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,7:9) = clustering_accuracy(grp,reshape(...
                res_grp_hat3(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,10:12) = clustering_accuracy(grp,reshape(...
                res_grp_hat4(simu,idxt,1:N),1,N));
            if min(N,T)>=10
                res_table2(simu,idxt,13:15) = clustering_accuracy(grp,reshape(...
                    res_grp_hat5(simu,idxt,1:N),1,N));
            end
            res_table2(simu,idxt,16:18) = clustering_accuracy(grp,reshape(...
                res_grp_hat6(simu,idxt,1:N),1,N));
            res_table2(simu,idxt,19:21) = clustering_accuracy(grp,reshape(...
                res_grp_hat7(simu,idxt,1:N),1,N));        
            res_table2(simu,idxt,22:24) = clustering_accuracy(grp,reshape(...
                res_grp_hat8(simu,idxt,1:N),1,N));  
           
                res_table2(simu,idxt,25:27) = clustering_accuracy(grp,reshape(...
                    res_grp_hat9(simu,idxt,1:N),1,N));
        end       
    end
    
    % Save raw results
    save(append(path,'_res_Ghat_TPWD_G',string(G),'.mat'),'res_Ghat_TPWD');
    save(append(path,'_res_Ghat_S2_G',string(G),'.mat'),'res_Ghat_S2');
    save(append(path,'_res_Ghat_S3_G',string(G),'.mat'),'res_Ghat_S3');
    save(append(path,'_res_Ghat_S4_G',string(G),'.mat'),'res_Ghat_S4');
    save(append(path,'_res_Ghat_S10_G',string(G),'.mat'),'res_Ghat_S10');

    save(append(path,'_res_CPU_G',string(G),'.mat'),'res_CPU');
    save(append(path,'_res_grp_hat1_G',string(G),'.mat'),'res_grp_hat1');
    save(append(path,'_res_grp_hat2_G',string(G),'.mat'),'res_grp_hat2');
    save(append(path,'_res_grp_hat3_G',string(G),'.mat'),'res_grp_hat3');
    save(append(path,'_res_grp_hat4_G',string(G),'.mat'),'res_grp_hat4');
    save(append(path,'_res_grp_hat5_G',string(G),'.mat'),'res_grp_hat5');
    save(append(path,'_res_grp_hat6_G',string(G),'.mat'),'res_grp_hat6');
    save(append(path,'_res_grp_hat7_G',string(G),'.mat'),'res_grp_hat7');
    save(append(path,'_res_grp_hat8_G',string(G),'.mat'),'res_grp_hat8');
    save(append(path,'_res_grp_hat9_G',string(G),'.mat'),'res_grp_hat9');
    save(append(path,'_res_grp_hat10_G',string(G),'.mat'),'res_grp_hat10');


    save(append(path,'_res_alpha_hat1_G',string(G),'.mat'),'res_alpha_hat1');
    save(append(path,'_res_alpha_hat2_G',string(G),'.mat'),'res_alpha_hat2');
    save(append(path,'_res_alpha_hat3_G',string(G),'.mat'),'res_alpha_hat3');
    save(append(path,'_res_alpha_hat4_G',string(G),'.mat'),'res_alpha_hat4');
    save(append(path,'_res_alpha_hat5_G',string(G),'.mat'),'res_alpha_hat5');
    save(append(path,'_res_alpha_hat6_G',string(G),'.mat'),'res_alpha_hat6');
    save(append(path,'_res_alpha_hat7_G',string(G),'.mat'),'res_alpha_hat7');
    save(append(path,'_res_alpha_hat8_G',string(G),'.mat'),'res_alpha_hat8');
    save(append(path,'_res_alpha_hat9_G',string(G),'.mat'),'res_alpha_hat9');
    save(append(path,'_res_alpha_hat10_G',string(G),'.mat'),'res_alpha_hat10');
    
    % Save raw Table 1 (Pure GFE: Ghat, RMSE and CPU time)
    save(append(path,'_res_table1_G',string(G),'.mat'),'res_table1');
    %save(append(path,'_res_tableS1_G',string(G),'.mat'),'res_table1');
    %save(append(path,'_res_tableS3_G',string(G),'.mat'),'res_table1');
    %save(append(path,'_sres_tableS5_G',string(G),'.mat'),'res_table1');

    % Output raw Table 2 (Pure GFE: classification accuracy)
    save(append(path,'_res_table2_G',string(G),'.mat'),'res_table2');
    %save(append(path,'_res_tableS2_G',string(G),'.mat'),'res_table2');
    %save(append(path,'_res_tableS4_G',string(G),'.mat'),'res_table2');
    %save(append(path,'_res_tableS6_G',string(G),'.mat'),'res_table2');
   if G==3
       res_full_tab1(:,1:8,:) = res_table1;
       res_full_tab2(:,1:8,:) = res_table2;
   else
       res_full_tab1(:,9:16,:) = res_table1;
       res_full_tab2(:,9:16,:) = res_table2;
   end
end

tab1_out = round(reshape(mean(res_full_tab1,1),16,16),3);
save(append(path,'_table1.mat'),'tab1_out');

tab2_out = round(reshape(mean(res_full_tab2,1),16,27),3);
save(append(path,'_table2.mat'),'tab2_out');

% Output LaTex Tables 1,S1,S3,S5
fileID = fopen('output_table1.tex','w');
%fileID = fopen('output_tableS1.tex','w'); %Mute if needed
%fileID = fopen('output_tableS3.tex','w'); %Mute if needed
%fileID = fopen('output_tableS5.tex','w'); %Mute if needed

fprintf(fileID, '\\begin{tabular}{lll *{25}{S[table-format=-1.3]}}\n');
fprintf(fileID, '\\toprule\n');
fprintf(fileID, '{} & {} & {} & \\multicolumn{3}{c}{TPWD} & & \\multicolumn{2}{c}{Post-Spectral$^{\\bar G=2}$} & & \\multicolumn{2}{c}{Post-Spectral$^{\\bar G=3}$} &&  \\multicolumn{2}{c}{Post-Spectral$^{\\bar G=4}$} & & \\multicolumn{2}{c}{Post-Spectral$^{\\bar G=10}$} & & \\multicolumn{1}{c}{GFE$^{\\bar G=2}$} & & \\multicolumn{1}{c}{GFE$^{\\bar G=3}$} & & \\multicolumn{1}{c}{GFE$^{\\bar G=4}$} &  & \\multicolumn{1}{c}{GFE$^{\\bar G=10}$} && \\multicolumn{1}{c}{Oracle} \\\\\n');
fprintf(fileID, '\\cmidrule{4-6}\\cmidrule{8-9}\\cmidrule{11-12}\\cmidrule{14-15}\\cmidrule{17-18}\n');
fprintf(fileID, '$G$ & $N$ & $T$ & \\multicolumn{1}{c}{$\\widehat G$} & \\multicolumn{1}{c}{RMSE} & \\multicolumn{1}{c}{CPU time} & & \\multicolumn{1}{c}{$\\widehat G$} & \\multicolumn{1}{c}{RMSE} && \\multicolumn{1}{c}{$\\widehat G$} & \\multicolumn{1}{c}{RMSE} && \\multicolumn{1}{c}{$\\widehat G$} & \\multicolumn{1}{c}{RMSE} && \\multicolumn{1}{c}{$\\widehat G$} & \\multicolumn{1}{c}{RMSE} && \\multicolumn{1}{c}{RMSE} && \\multicolumn{1}{c}{RMSE} && \\multicolumn{1}{c}{RMSE}  && \\multicolumn{1}{c}{RMSE} && \\multicolumn{1}{c}{RMSE}  \\\\\n');
fprintf(fileID, '\\midrule \n');
i = 1;
for G=[3,4]
    for N=[90,180]
        for T=Tseq
            if G==3
                if T==7 
                    fprintf(fileID, string(G)+'&'+string(N)+'&'+string(T)+'&\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & & \\cellcolor{mygreen!25}%.3f & \\cellcolor{mygreen!25}%.3f & & %.3f & %.3f & & %.3f & %.3f & & %.3f & & \\cellcolor{mygreen!25} %.3f & & %.3f & & %.3f && %.3f  \\\\ \n', tab1_out(i, :));
                else
                    fprintf(fileID,'& &'+string(T)+'&\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & & \\cellcolor{mygreen!25}%.3f & \\cellcolor{mygreen!25}%.3f & & %.3f & %.3f & & %.3f & %.3f & & %.3f & & \\cellcolor{mygreen!25} %.3f & & %.3f & & %.3f && %.3f  \\\\ \n', tab1_out(i, :));
                end
            else
                if T==7 
                    fprintf(fileID, string(G)+'&'+string(N)+'&'+string(T)+'&\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & & %.3f & %.3f & & \\cellcolor{mygreen!25}%.3f & \\cellcolor{mygreen!25}%.3f & & %.3f & %.3f & & %.3f & & %.3f & & \\cellcolor{mygreen!25} %.3f & & %.3f & &  %.3f  \\\\ \n', tab1_out(i, :));
                else
                    fprintf(fileID,'& &'+string(T)+'&\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f &\\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & & %.3f & %.3f & & \\cellcolor{mygreen!25}%.3f & \\cellcolor{mygreen!25}%.3f & & %.3f & %.3f & & %.3f & & %.3f & & \\cellcolor{mygreen!25} %.3f & & %.3f & & %.3f  \\\\ \n', tab1_out(i, :));
                end
            end    
            i = i+1;
        end
        if G==3 || N==90
            fprintf(fileID, '\\midrule\n');
        end

    end
end
fprintf(fileID, '\\bottomrule\n');
fprintf(fileID, '\\end{tabular}\n');
fclose(fileID);

% Output LaTex Tables 2,S2,S4,S6
fileID = fopen('output_table2.tex','w');
%fileID = fopen('output_tableS2.tex','w'); %Mute if needed
%fileID = fopen('output_tableS4.tex','w'); %Mute if needed
%fileID = fopen('output_tableS6.tex','w'); %Mute if needed

fprintf(fileID, '\\begin{tabular}{lll *{35}{S[table-format=-1.3]}}\n');
fprintf(fileID, '\\toprule\n');
fprintf(fileID, '{} & {} & {} & \\multicolumn{3}{c}{TPWD} & & \\multicolumn{3}{c}{Post-Spectral$^{\\bar G=2}$} & & \\multicolumn{3}{c}{Post-Spectral$^{\\bar G=3}$} & & \\multicolumn{3}{c}{Post-Spectral$^{\\bar G=4}$} & & \\multicolumn{3}{c}{Post-Spectral$^{\\bar G=10}$} & & \\multicolumn{3}{c}{GFE$^{\\bar G=2}$} & & \\multicolumn{3}{c}{GFE$^{\\bar G=3}$} & & \\multicolumn{3}{c}{GFE$^{\\bar G=4}$} & & \\multicolumn{3}{c}{GFE$^{\\bar G=10}$}  \\\\\n');
fprintf(fileID, '\\cmidrule{4-6}\\cmidrule{8-10}\\cmidrule{12-14}\\cmidrule{16-18}\\cmidrule{20-22}\\cmidrule{24-26}\\cmidrule{28-30}\\cmidrule{32-34}\\cmidrule{36-38}\n');
fprintf(fileID, '$G$ &$N$ & $T$ &  P & R &  RI && P &   R &  RI && P &   R &  RI && P &   R &  RI && P &   R &  RI && P &   R &  RI && P &   R &  RI && P &   R &  RI  && P &   R &  RI\\\\\n');
fprintf(fileID, '\\midrule \n');
i = 1;
for G=[3,4]
    for N=[90,180]
        for T=Tseq
            if G==3
                if T==7 
                    fprintf(fileID, string(G)+'&'+string(N)+'&'+string(T)+'& \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f  \\\\ \n', tab2_out(i, :));
                else
                    fprintf(fileID,'& &'+string(T)+'& \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f  \\\\ \n', tab2_out(i, :));
                end
            else
                if T==7 
                    fprintf(fileID, string(G)+'&'+string(N)+'&'+string(T)+'& \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f   \\\\ \n', tab2_out(i, :));
                else
                    fprintf(fileID,'& &'+string(T)+'& \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & \\cellcolor{mygreen!25} %.3f & & %.3f & %.3f & %.3f   \\\\ \n', tab2_out(i, :));
                end
            end
            i = i+1;
        end
        if G==3 || N==90
            fprintf(fileID, '\\midrule\n');
        end
    end
end
fprintf(fileID, '\\bottomrule\n');
fprintf(fileID, '\\end{tabular}\n');
fclose(fileID);