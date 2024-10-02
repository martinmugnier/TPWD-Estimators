%--------------------- MONTE CARLO SIMULATIONS ----------------------------
% This code produces Tables:
%  - 3,4: iid errors, signal-to-noise ratio=1, average link;
%  - S7, S8: correlated errors, heteroscedasticity, 
%             signal-to-noise ratio=1, average link;
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
% Mugnier, M. (2022), A Simple and Computationally Trivial Estimator for 
% Grouped Fixed Effects Models.
%--------------------------------------------------------------------------

clear;

rng('default');

SM_tables = false; %set to false to output Tables 3,4 (true for S7 & S8)

% Settings
K = 1;
beta = ones(K,1);
balanced_grp = 1; 
error_correlation = SM_tables;
sigma = 1/3; 

Gseq = [3,4];
Tseq = [7,10,20,40];

% BM (2015)'s GFE estimator
n_init_GFE = 100;
% Bai (2009)'s IFE estimator
n_init_IFE = 100; 
% Bonhomme and Manresa (2015)'s GFE estimator
Gmax = 5; 
% Chetverikov and Manresa (2022)'s spectral and post-spectral estimators
spectr_lbda_grid = linspace(1,50,100); 
M = 1; 
% Mugnier (2022)'s TPWD estimator
link = 'average'; 

% Path to send results
path = append('tables_3_and_4/',link);
if SM_tables
    path = append('tables_3_and_4/',link,'heteroscedasticity_correlation');
end
%path = append('tables_3_and_4/',link,'_squared');

%Simulations
ncores = 40;
nsim = 500;

res_full_tab3 = nan(nsim,16,46);
res_full_tab4 = nan(nsim,16,15);

for G=Gseq
    res_beta_hat1 = zeros(nsim,8,K);  % TPWD 
    res_beta_hat2 = zeros(nsim,8,K);  % TPWD 4 iterations 
    res_beta_hat3 = zeros(nsim,8,K);  % NNR 
    res_beta_hat4 = zeros(nsim,8,K);  % NN 
    res_beta_hat5 = zeros(nsim,8,K);  % Spectral (not feasible if N<2GM+2) 
    res_beta_hat6 = zeros(nsim,8,K);  % Post-spectral (not feasible if T<10) 
    res_beta_hat7 = zeros(nsim,8,K);  % GFE 
    res_beta_hat8 = zeros(nsim,8,K);  % GFE with BIC selection of G 
    res_beta_hat9 = zeros(nsim,8,K);  % IFE random init 
    res_beta_hat10 = zeros(nsim,8,K); % IFE random init, bias-corrected 
    res_beta_hat11 = zeros(nsim,8,K); % IFE init at NNR 
    res_beta_hat12 = zeros(nsim,8,K); % IFE init at NNR, bias-corrected 
    res_beta_hat13 = zeros(nsim,8,K); % Oracle 
    
    res_var_beta_hat1 = zeros(nsim,8,K,K);
    res_var_beta_hat2 = zeros(nsim,8,K,K);
    res_var_beta_hat6 = zeros(nsim,8,K,K);
    res_var_beta_hat7 = zeros(nsim,8,K,K);
    res_var_beta_hat8 = zeros(nsim,8,K,K);
    res_var_beta_hat9 = zeros(nsim,8,K,K);
    res_var_beta_hat10 = zeros(nsim,8,K,K);
    res_var_beta_hat11 = zeros(nsim,8,K,K);
    res_var_beta_hat12 = zeros(nsim,8,K,K);
    res_var_beta_hat13 = zeros(nsim,8,K,K);
    
    res_G_hat1 = zeros(nsim,8);    
    res_G_hat2 = zeros(nsim,8); 
    res_G_hat6 = zeros(nsim,8);      
    res_G_hat8 = zeros(nsim,8); 

    res_grp_hat1 = zeros(nsim,8,180);  
    res_grp_hat2 = zeros(nsim,8,180); 
    res_grp_hat6 = zeros(nsim,8,180);  
    res_grp_hat7 = zeros(nsim,8,180); 
    res_grp_hat8 = zeros(nsim,8,180);  
    res_grp_hat13 = zeros(nsim,8,180); 

    res_alpha_hat1 = zeros(nsim,8,180,40);
    res_alpha_hat2 = zeros(nsim,8,180,40);
    res_alpha_hat6 = zeros(nsim,8,180,40);
    res_alpha_hat7 = zeros(nsim,8,180,40);
    res_alpha_hat8 = zeros(nsim,8,180,40);
    res_alpha_hat13 = zeros(nsim,8,180,40); 

    res_table3 = nan(nsim,8,46); 
    res_table4 = nan(nsim,8,15); 
                  
    res_loop_var_beta_hat1 = zeros(nsim,K,K);
    res_loop_var_beta_hat2 = zeros(nsim,K,K);
    res_loop_var_beta_hat6 = zeros(nsim,K,K);
    res_loop_var_beta_hat7 = zeros(nsim,K,K);
    res_loop_var_beta_hat8 = zeros(nsim,K,K);
    res_loop_var_beta_hat9 = zeros(nsim,K,K);
    res_loop_var_beta_hat10 = zeros(nsim,K,K);
    res_loop_var_beta_hat11 = zeros(nsim,K,K);
    res_loop_var_beta_hat12 = zeros(nsim,K,K);
    res_loop_var_beta_hat13 = zeros(nsim,K,K);
    
    res_loop_alpha_hat1 = zeros(nsim,180,40);
    res_loop_alpha_hat2 = zeros(nsim,180,40);
    res_loop_alpha_hat6 = zeros(nsim,180,40);
    res_loop_alpha_hat7 = zeros(nsim,180,40);
    res_loop_alpha_hat8 = zeros(nsim,180,40);
    res_loop_alpha_hat13 = zeros(nsim,180,40);

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
        err = zeros(N,T,nsim);
        if error_correlation==1
            for simu=1:nsim
                    noise_mod = arima('Constant',0,'AR',{0.20},...
                        'Variance',sigma^2); 
                    err(:,:,simu) = simulate(noise_mod,T,'NumPaths',N)'; 
            end
        else
            err = normrnd(0,sigma,N,T,nsim);
        end
        grp_effects_mat = grp_dum_mat*alpha;
        Xmat = normrnd(0,sigma,N,T,K,nsim)+0.5*grp_effects_mat;
        res_loop_grp_hat1 = zeros(nsim,N);
        res_loop_grp_hat2 = zeros(nsim,N);
        res_loop_grp_hat6 = zeros(nsim,N);
        res_loop_grp_hat7 = zeros(nsim,N);
        res_loop_grp_hat8 = zeros(nsim,N);
        res_loop_grp_hat13 = zeros(nsim,N);
        
        res_loop_beta_hat1 = zeros(nsim,K);
        res_loop_beta_hat2 = zeros(nsim,K);
        res_loop_beta_hat3 = zeros(nsim,K);
        res_loop_beta_hat4 = zeros(nsim,K);
        res_loop_beta_hat5 = zeros(nsim,K);
        res_loop_beta_hat6 = zeros(nsim,K);
        res_loop_beta_hat7 = zeros(nsim,K);
        res_loop_beta_hat8 = zeros(nsim,K);
        res_loop_beta_hat9 = zeros(nsim,K);
        res_loop_beta_hat10 = zeros(nsim,K);
        res_loop_beta_hat11 = zeros(nsim,K);
        res_loop_beta_hat12 = zeros(nsim,K);
        res_loop_beta_hat13 = zeros(nsim,K);
        
        parfor (simu=1:nsim,ncores)
        %for simu=1:nsim
            disp([N,idxt,simu]);
            X = reshape(Xmat(:,:,:,simu),N,T,K);
            Y = sum(X.*reshape(kron(beta',ones([N,T])),N,T,K),3)+...
                grp_effects_mat+...
                (1+SM_tables.*sum(X.*reshape(kron(beta',...
                ones([N,T])),N,T,K),3)).*squeeze(err(:,:,simu));
            % Step 1: Compute TPWD preliminary estimators
            psi_nnr = log(log(T))/(4*sqrt(min(N,T)));
            res_loop_beta_hat3(simu,:) = MW_nucnormreg_estimator_optim(...
                Y,X,psi_nnr); 
            res_loop_beta_hat4(simu,:) = MW_nucnorm_estimator(Y,X);
            % Compute TPWD estimator (1 iteration)
            % Step 2: Classification
            residuals = Y-sum(X.*reshape(kron(res_loop_beta_hat3(simu,...
                :)',ones([N,T])),N,T,K),3);
            sigma_hat = std(residuals,0,'all'); 
            c = 1.5*sigma_hat^2*log(T)*T^(-1/2);
            [res_G_hat1(simu,idxt),res_loop_grp_hat1(simu,:)] = ...
                TPWD_clustering(residuals,c,link);
            % Step 3: Linear projection
            [res_loop_beta_hat1(simu,:),alpha_hat1] = FE_reg_withcov(Y,...
                X,res_loop_grp_hat1(simu,:)');
            [var_beta_hat1,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat1(simu,:)',alpha_hat1,...
                  res_loop_grp_hat1(simu,:)');
            % Compute TPWD estimator with 4 iterations
            beta_init = res_loop_beta_hat1(simu,:)';
            for j=1:3
                % Clustering
                residuals = Y-sum(X.*reshape(kron(...
                      beta_init',ones([N,T])),N,T,K),3);
                sigma_hat = std(residuals,0,'all'); 
                c = 1.5*sigma_hat^2*log(T)*T^(-1/2);
                [res_G_hat2(simu,idxt),res_loop_grp_hat2(simu,:)] ...
                      = TPWD_clustering(residuals,c,link);
                % Linear projection
                [res_loop_beta_hat2(simu,:),alpha_hat2] = ...
                      FE_reg_withcov(Y,X,res_loop_grp_hat2(simu,:)');
                beta_init = res_loop_beta_hat2(simu,:)';
            end
            [var_beta_hat2,~] = compute_GFE_analytical_cov(Y,X,...
                 res_loop_beta_hat2(simu,:)',alpha_hat2,...
                  res_loop_grp_hat2(simu,:)');
            % Compute spectral and post-spectral estimators
            res_loop_beta_hat5(simu,:) = CM_spectral_estimator(Y,X,G*M);
            [res_loop_beta_hat6(simu,:), alpha_hat6, ...
                 res_loop_grp_hat6(simu,:)] = CM_post_spectral_estimator(...
                 Y,X,G,M,spectr_lbda_grid);
            res_G_hat6(simu,idxt) = max(res_loop_grp_hat6(simu,:));
            [var_beta_hat6,~] = compute_GFE_analytical_cov(Y,X,...
                 res_loop_beta_hat6(simu,:)',alpha_hat6,...
                 res_loop_grp_hat6(simu,:)');
            % Compute GFE estimator (Algo 1)
            [res_loop_beta_hat7(simu,:), alpha_hat7,...
                  res_loop_grp_hat7(simu,:),objAlgo1] = ...
                  BM_algo1_multiple_init(Y,X,G,n_init_GFE);
            [var_beta_hat7,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat7(simu,:)',alpha_hat7,...
                  res_loop_grp_hat7(simu,:)');
            % Compute GFE estimator (BIC crit. to select groups with Gmax=5)
            objBIC = zeros(Gmax,1);
            objBIC(G) = objAlgo1;
            betaBIC = zeros(Gmax,K);
            betaBIC(G,:) = res_loop_beta_hat7(simu,:)';
            alphaBIC = zeros(Gmax,Gmax,T);
            alphaBIC(G,1:G,:) = alpha_hat7;
            grpBIC = zeros(Gmax,N);
            grpBIC(G,:) = res_loop_grp_hat7(simu,:)';
            [betaBIC(Gmax,:),alphaBIC(Gmax,1:Gmax,:),...
                  grpBIC(Gmax,:),objBIC(Gmax)] = BM_algo1_multiple_init(...
                  Y,X,Gmax,n_init_GFE);
            G_BIC_list = 1:Gmax-1;
            G_BIC_list = G_BIC_list(G_BIC_list~=G);
            for j=G_BIC_list
                [betaBIC(j,:),alphaBIC(j,1:j,:),...
                  grpBIC(j,:),objBIC(j)] = BM_algo1_multiple_init(Y,X,j,...
                  n_init_GFE);
            end
            sig_sq_BIC = objBIC(Gmax)/(N*T-Gmax*T-N-K);
            objBIC = objBIC/(N*T) + ...
                reshape(sig_sq_BIC*((1:Gmax)*T+N+K)/(N*T)*log(N*T),Gmax,1);
            [~,Gstar] = min(objBIC);
            res_loop_beta_hat8(simu,:) = betaBIC(Gstar,:)';
            alpha_hat8 = squeeze(alphaBIC(Gstar,1:Gstar,:));
            res_loop_grp_hat8(simu,:) = grpBIC(Gstar,:)';
            [var_beta_hat8,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat8(simu,:)',alpha_hat8,...
                  res_loop_grp_hat8(simu,:)');
            res_G_hat8(simu,idxt) = Gstar; 
            % Compute interactive fixed effects estimators 
            % (random init, bias corrected)
            XX = zeros(K,N,T);
            for k=1:K
                XX(k,:,:) = X(:,:,k);
            end
            stop = false;
            fail = 0;
            while ~stop
                try
                    [res_loop_beta_hat9(simu,:),~,~,~,~,var_beta_hat9,~,...
                        B1,B2,B3] = LS_factor(Y,XX,G,'silent', 1e-8,'m2',...
                        zeros(K,1),n_init_IFE,n_init_IFE);
                    stop = true;
                catch
                    fail = fail + 1;
                    disp(string(fail)+...
               ' failures IFE random init, 500 it. (simu, G, N, T) =  ('...
                          +string(simu)+','+string(G)+','+string(N)+','...
                          +string(T)+').');
                end
            end
            res_loop_beta_hat10(simu,:) = res_loop_beta_hat9(simu,:)'...
                -B1-B2-B3;
            var_beta_hat10 = var_beta_hat9;
            % Compute interactive fixed effects estimators 
            %(init at beta_nnr, bias corrected)
            stop = false;
            fail = 0;
            while ~stop
                try
                    [res_loop_beta_hat11(simu,:),~,~,~,~,var_beta_hat11,...
                        ~,B1,B2,B3] = LS_factor(Y,XX,G,'silent',1e-8,...
                        'm1',res_loop_beta_hat3(simu,:)',1,1);
                    stop = true;
                 catch
                    fail = fail + 1;
                    disp(string(fail)+...
                        ' failures NNR random init (simu, G, N, T) =  ('...
                        +string(simu)+','+string(G)+','+string(N)+','...
                        +string(T)+').');
                end
            end
            res_loop_beta_hat12(simu,:) = res_loop_beta_hat11(simu,:)'...
                -B1-B2-B3;
            var_beta_hat12 = var_beta_hat11;
            % Compute oracle pooled OLS estimator
            [res_loop_beta_hat13(simu,:), alpha_hat13] = ...
                  FE_reg_withcov(Y,X,grp);
            [var_beta_hat13,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat13(simu,:)',alpha_hat13,grp);
            res_loop_grp_hat13(simu,:) = grp;
            
            % handle arrays of changing size in parfor loop
            alpha_hat1_pad = zeros(180,40);
            alpha_hat2_pad = zeros(180,40);
            alpha_hat6_pad = zeros(180,40);
            alpha_hat7_pad = zeros(180,40);
            alpha_hat8_pad = zeros(180,40);
            alpha_hat13_pad = zeros(180,40);

            alpha_hat1_pad(1:res_G_hat1(simu,idxt),1:T) = alpha_hat1;
            alpha_hat2_pad(1:res_G_hat2(simu,idxt),1:T) = alpha_hat2;
            alpha_hat6_pad(1:res_G_hat6(simu,idxt),1:T) = alpha_hat6;
            alpha_hat7_pad(1:G,1:T) = alpha_hat7;
            alpha_hat8_pad(1:res_G_hat8(simu,idxt),1:T) = alpha_hat8;
            alpha_hat13_pad(1:G,1:T) = alpha_hat13;

            res_loop_alpha_hat1(simu,:,:) = alpha_hat1_pad;
            res_loop_alpha_hat2(simu,:,:) = alpha_hat2_pad;
            res_loop_alpha_hat6(simu,:,:) = alpha_hat6_pad;
            res_loop_alpha_hat7(simu,:,:) = alpha_hat7_pad;
            res_loop_alpha_hat8(simu,:,:) = alpha_hat8_pad;
            res_loop_alpha_hat13(simu,:,:) = alpha_hat13_pad;
            
            res_loop_var_beta_hat1(simu,:,:) = var_beta_hat1;
            res_loop_var_beta_hat2(simu,:,:) = var_beta_hat2;
            res_loop_var_beta_hat6(simu,:,:) = var_beta_hat6;
            res_loop_var_beta_hat7(simu,:,:) = var_beta_hat7;
            res_loop_var_beta_hat8(simu,:,:) = var_beta_hat8;
            res_loop_var_beta_hat9(simu,:,:) = var_beta_hat9;
            res_loop_var_beta_hat10(simu,:,:) = var_beta_hat10;
            res_loop_var_beta_hat11(simu,:,:) = var_beta_hat11;
            res_loop_var_beta_hat12(simu,:,:) = var_beta_hat12;
            res_loop_var_beta_hat13(simu,:,:) = var_beta_hat13;


        end

        % store results and fill in table
        for simu=1:nsim
            res_grp_hat1(simu,idxt,1:N) = res_loop_grp_hat1(simu,:);
            res_grp_hat2(simu,idxt,1:N) = res_loop_grp_hat2(simu,:);
            res_grp_hat6(simu,idxt,1:N) = res_loop_grp_hat6(simu,:);
            res_grp_hat7(simu,idxt,1:N) = res_loop_grp_hat7(simu,:);
            res_grp_hat8(simu,idxt,1:N) = res_loop_grp_hat8(simu,:);
            res_grp_hat13(simu,idxt,1:N) = res_loop_grp_hat13(simu,:);
            
            res_beta_hat1(simu,idxt,:) = res_loop_beta_hat1(simu,:);
            res_beta_hat2(simu,idxt,:) = res_loop_beta_hat2(simu,:);
            res_beta_hat3(simu,idxt,:) = res_loop_beta_hat3(simu,:);
            res_beta_hat4(simu,idxt,:) = res_loop_beta_hat4(simu,:);
            res_beta_hat5(simu,idxt,:) = res_loop_beta_hat5(simu,:);
            res_beta_hat6(simu,idxt,:) = res_loop_beta_hat6(simu,:);
            res_beta_hat7(simu,idxt,:) = res_loop_beta_hat7(simu,:);
            res_beta_hat8(simu,idxt,:) = res_loop_beta_hat8(simu,:);
            res_beta_hat9(simu,idxt,:) = res_loop_beta_hat9(simu,:);
            res_beta_hat10(simu,idxt,:) = res_loop_beta_hat10(simu,:);
            res_beta_hat11(simu,idxt,:) = res_loop_beta_hat11(simu,:);
            res_beta_hat12(simu,idxt,:) = res_loop_beta_hat12(simu,:);
            res_beta_hat13(simu,idxt,:) = res_loop_beta_hat13(simu,:);
            
            res_var_beta_hat1(simu,idxt,:,:) = ...
                res_loop_var_beta_hat1(simu,:,:);
            res_var_beta_hat2(simu,idxt,:,:) = ...
                res_loop_var_beta_hat2(simu,:,:);
            res_var_beta_hat6(simu,idxt,:,:) = ...
                res_loop_var_beta_hat6(simu,:,:);
            res_var_beta_hat7(simu,idxt,:,:) = ...
                res_loop_var_beta_hat7(simu,:,:);
            res_var_beta_hat8(simu,idxt,:,:) = ...
                res_loop_var_beta_hat8(simu,:,:);
            res_var_beta_hat9(simu,idxt,:,:) = ...
                res_loop_var_beta_hat9(simu,:,:);
            res_var_beta_hat10(simu,idxt,:,:) = ...
                res_loop_var_beta_hat10(simu,:,:);
            res_var_beta_hat11(simu,idxt,:,:) = ...
                res_loop_var_beta_hat11(simu,:,:);
            res_var_beta_hat12(simu,idxt,:,:) = ...
                res_loop_var_beta_hat12(simu,:,:);
            res_var_beta_hat13(simu,idxt,:,:) = ...
                res_loop_var_beta_hat13(simu,:,:);

            res_alpha_hat1(simu,idxt,:,:) = res_loop_alpha_hat1(simu,:,:);
            res_alpha_hat2(simu,idxt,:,:) = res_loop_alpha_hat2(simu,:,:);
            res_alpha_hat6(simu,idxt,:,:) = res_loop_alpha_hat6(simu,:,:);
            res_alpha_hat7(simu,idxt,:,:) = res_loop_alpha_hat7(simu,:,:);
            res_alpha_hat8(simu,idxt,:,:) = res_loop_alpha_hat8(simu,:,:);
            res_alpha_hat13(simu,idxt,:,:) = res_loop_alpha_hat13(simu,:,:);

            res_table3(simu,idxt,1) = res_beta_hat1(simu,idxt,:)-beta;
            res_table3(simu,idxt,2) = mean((res_beta_hat1(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat1(simu,idxt,1,1));
            res_table3(simu,idxt,3) = (beta>=res_beta_hat1(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat1(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,4) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat1(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat1(simu,idxt,...
                1:res_G_hat1(simu,idxt),1:T),...
                res_G_hat1(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,5) = res_G_hat1(simu,idxt);

            res_table3(simu,idxt,6) = res_beta_hat2(simu,idxt,:)-beta;
            res_table3(simu,idxt,7) = mean((res_beta_hat2(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat2(simu,idxt,1,1));
            res_table3(simu,idxt,8) = (beta>=res_beta_hat2(simu,idxt,:)...
               -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat2(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,9) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat2(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat2(simu,idxt,1:res_G_hat2(simu,...
                idxt),1:T),res_G_hat2(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,10) = res_G_hat2(simu,idxt);
            
            res_table3(simu,idxt,11) = res_beta_hat3(simu,idxt,:)-beta;
            res_table3(simu,idxt,12) = mean((res_beta_hat3(simu,idxt,:)...
                -beta).^2);
            
            res_table3(simu,idxt,13) = res_beta_hat4(simu,idxt,:)-beta;
            res_table3(simu,idxt,14) = mean((res_beta_hat4(simu,idxt,:)...
                -beta).^2);

            res_table3(simu,idxt,15) = res_beta_hat5(simu,idxt,:)-beta;
            res_table3(simu,idxt,16) = mean((res_beta_hat5(simu,idxt,:)...
                -beta).^2);
            
            res_table3(simu,idxt,17) = res_beta_hat6(simu,idxt,:)-beta;
            res_table3(simu,idxt,18) = mean((res_beta_hat6(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat6(simu,idxt,1,1));
            res_table3(simu,idxt,19) = (beta>=res_beta_hat6(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat6(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,20) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat6(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat6(simu,idxt,1:res_G_hat6(simu,...
                idxt),1:T),res_G_hat6(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,21) = res_G_hat6(simu,idxt);
            
            res_table3(simu,idxt,22) = res_beta_hat7(simu,idxt,:)-beta;
            res_table3(simu,idxt,23) = mean((res_beta_hat7(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat7(simu,idxt,1,1));
            res_table3(simu,idxt,24) = (beta>=res_beta_hat7(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat7(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,25) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat7(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat7(simu,idxt,1:G,1:T),G,T)).^2,'all'));
            
            res_table3(simu,idxt,26) = res_beta_hat8(simu,idxt,:)-beta;
            res_table3(simu,idxt,27) = mean((res_beta_hat8(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat8(simu,idxt,1,1));
            res_table3(simu,idxt,28) = (beta>=res_beta_hat8(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat8(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,29) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat8(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat8(simu,idxt,...
                1:res_G_hat8(simu,idxt),1:T),...
                res_G_hat8(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,30) = res_G_hat8(simu,idxt);
            
            res_table3(simu,idxt,31) = res_beta_hat9(simu,idxt,:)-beta;
            res_table3(simu,idxt,32) = mean((res_beta_hat9(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat9(simu,idxt,1,1));
            res_table3(simu,idxt,33) = (beta>=res_beta_hat9(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat9(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,34) = res_beta_hat10(simu,idxt,:)-beta;
            res_table3(simu,idxt,35) = mean((res_beta_hat10(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat10(simu,idxt,1,1));
            res_table3(simu,idxt,36) = (beta>=res_beta_hat10(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat10(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,37) = res_beta_hat11(simu,idxt,:)-beta;
            res_table3(simu,idxt,38) = mean((res_beta_hat11(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat11(simu,idxt,1,1));
            res_table3(simu,idxt,39) = (beta>=res_beta_hat11(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat11(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,40) = res_beta_hat12(simu,idxt,:)-beta;
            res_table3(simu,idxt,41) = mean((res_beta_hat12(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat12(simu,idxt,1,1));
            res_table3(simu,idxt,42) = (beta>=res_beta_hat12(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat12(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,43) = res_beta_hat13(simu,idxt,:)-beta;
            res_table3(simu,idxt,44) = mean((res_beta_hat13(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat13(simu,idxt,1,1));
            res_table3(simu,idxt,45) = (beta>=res_beta_hat13(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat13(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,46) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat13(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat13(simu,idxt,1:G,1:T),...
                G,T)).^2,'all'));
            
            res_table4(simu,idxt,1:3) = clustering_accuracy(grp, ...
                squeeze(res_grp_hat1(simu,idxt,1:N)));
            res_table4(simu,idxt,4:6) = clustering_accuracy(grp,...
                squeeze(res_grp_hat2(simu,idxt,1:N)));
            res_table4(simu,idxt,7:9) = clustering_accuracy(grp,...
                squeeze(res_grp_hat6(simu,idxt,1:N)));
            res_table4(simu,idxt,10:12) = clustering_accuracy(grp,...
                squeeze(res_grp_hat7(simu,idxt,1:N)));       
            res_table4(simu,idxt,13:15) = clustering_accuracy(grp,...
                squeeze(res_grp_hat8(simu,idxt,1:N))); 
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
                err = zeros(N,T,nsim);
        if error_correlation==1
            for simu=1:nsim
                    noise_mod = arima('Constant',0,'AR',{0.20},...
                        'Variance',sigma^2); 
                    err(:,:,simu) = simulate(noise_mod,T,'NumPaths',N)'; 
            end
        else
            err = normrnd(0,sigma,N,T,nsim);
        end
        grp_effects_mat = grp_dum_mat*alpha;
        Xmat = normrnd(0,sigma,N,T,K,nsim)+0.5*grp_effects_mat;
        
        res_loop_grp_hat1 = zeros(nsim,N);
        res_loop_grp_hat2 = zeros(nsim,N);
        res_loop_grp_hat6 = zeros(nsim,N);
        res_loop_grp_hat7 = zeros(nsim,N);
        res_loop_grp_hat8 = zeros(nsim,N);
        res_loop_grp_hat13 = zeros(nsim,N);
        
        res_loop_beta_hat1 = zeros(nsim,K);
        res_loop_beta_hat2 = zeros(nsim,K);
        res_loop_beta_hat3 = zeros(nsim,K);
        res_loop_beta_hat4 = zeros(nsim,K);
        res_loop_beta_hat5 = zeros(nsim,K);
        res_loop_beta_hat6 = zeros(nsim,K);
        res_loop_beta_hat7 = zeros(nsim,K);
        res_loop_beta_hat8 = zeros(nsim,K);
        res_loop_beta_hat9 = zeros(nsim,K);
        res_loop_beta_hat10 = zeros(nsim,K);
        res_loop_beta_hat11 = zeros(nsim,K);
        res_loop_beta_hat12 = zeros(nsim,K);
        res_loop_beta_hat13 = zeros(nsim,K);
        
        parfor (simu=1:nsim,ncores)
        %for simu=1:nsim
            disp([N,idxt,simu]);
            X = reshape(Xmat(:,:,:,simu),N,T,K);
            Y = sum(X.*reshape(kron(beta',ones([N,T])),N,T,K),3)+...
                grp_effects_mat+...
                (1+SM_tables.*sum(X.*reshape(kron(beta',...
                ones([N,T])),N,T,K),3)).*squeeze(err(:,:,simu));
            % Step 1: Compute TPWD preliminary estimators
            psi_nnr = log(log(T))/(4*sqrt(min(N,T)));
            res_loop_beta_hat3(simu,:) = MW_nucnormreg_estimator_optim(...
                Y,X,psi_nnr); 
            res_loop_beta_hat4(simu,:) = MW_nucnorm_estimator(Y,X);
            % Compute TPWD estimator (1 iteration)
            % Step 2: Classification
            residuals = Y-sum(X.*reshape(kron(res_loop_beta_hat3(simu,...
                :)',ones([N,T])),N,T,K),3);
            sigma_hat = std(residuals,0,'all'); 
            c = 1.5*sigma_hat^2*log(T)*T^(-1/2);
            [res_G_hat1(simu,idxt),res_loop_grp_hat1(simu,:)] = ...
                TPWD_clustering(residuals,c,link);
            % Step 3: Linear projection
            [res_loop_beta_hat1(simu,:),alpha_hat1] = FE_reg_withcov(Y,...
                X,res_loop_grp_hat1(simu,:)');
            [var_beta_hat1,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat1(simu,:)',alpha_hat1,...
                  res_loop_grp_hat1(simu,:)');
            % Compute TPWD estimator with 4 iterations
            beta_init = res_loop_beta_hat1(simu,:)';
            for j=1:3
                % Clustering
                residuals = Y-sum(X.*reshape(kron(...
                      beta_init',ones([N,T])),N,T,K),3);
                sigma_hat = std(residuals,0,'all'); 
                c = 1.5*sigma_hat^2*log(T)*T^(-1/2);
                [res_G_hat2(simu,idxt),res_loop_grp_hat2(simu,:)] ...
                      = TPWD_clustering(residuals,c,link);
                % Linear projection
                [res_loop_beta_hat2(simu,:),alpha_hat2] = ...
                      FE_reg_withcov(Y,X,res_loop_grp_hat2(simu,:)');
                beta_init = res_loop_beta_hat2(simu,:)';
            end
            [var_beta_hat2,~] = compute_GFE_analytical_cov(Y,X,...
                 res_loop_beta_hat2(simu,:)',alpha_hat2,...
                  res_loop_grp_hat2(simu,:)');
            % Compute spectral and post-spectral estimators
            res_loop_beta_hat5(simu,:) = CM_spectral_estimator(Y,X,G*M);
            [res_loop_beta_hat6(simu,:), alpha_hat6, ...
                 res_loop_grp_hat6(simu,:)] = CM_post_spectral_estimator(...
                 Y,X,G,M,spectr_lbda_grid);
            res_G_hat6(simu,idxt) = max(res_loop_grp_hat6(simu,:));
            [var_beta_hat6,~] = compute_GFE_analytical_cov(Y,X,...
                 res_loop_beta_hat6(simu,:)',alpha_hat6,...
                 res_loop_grp_hat6(simu,:)');
            % Compute GFE estimator (Algo 1)
            [res_loop_beta_hat7(simu,:), alpha_hat7,...
                  res_loop_grp_hat7(simu,:),objAlgo1] = ...
                  BM_algo1_multiple_init(Y,X,G,n_init_GFE);
            [var_beta_hat7,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat7(simu,:)',alpha_hat7,...
                  res_loop_grp_hat7(simu,:)');
            % Compute GFE estimator (BIC crit. to select groups with Gmax=5)
            objBIC = zeros(Gmax,1);
            objBIC(G) = objAlgo1;
            betaBIC = zeros(Gmax,K);
            betaBIC(G,:) = res_loop_beta_hat7(simu,:)';
            alphaBIC = zeros(Gmax,Gmax,T);
            alphaBIC(G,1:G,:) = alpha_hat7;
            grpBIC = zeros(Gmax,N);
            grpBIC(G,:) = res_loop_grp_hat7(simu,:)';
            [betaBIC(Gmax,:),alphaBIC(Gmax,1:Gmax,:),...
                  grpBIC(Gmax,:),objBIC(Gmax)] = BM_algo1_multiple_init(...
                  Y,X,Gmax,n_init_GFE);
            G_BIC_list = 1:Gmax-1;
            G_BIC_list = G_BIC_list(G_BIC_list~=G);
            for j=G_BIC_list
                [betaBIC(j,:),alphaBIC(j,1:j,:),...
                  grpBIC(j,:),objBIC(j)] = BM_algo1_multiple_init(Y,X,j,...
                  n_init_GFE);
            end
            sig_sq_BIC = (N*T*objBIC(Gmax))/(N*T-Gmax*T-N-K);
            objBIC = objBIC + ...
                reshape(sig_sq_BIC*((1:Gmax)*T+N+K)/(N*T)*log(N*T),Gmax,1);
            [~,Gstar] = min(objBIC);
            res_loop_beta_hat8(simu,:) = betaBIC(Gstar,:)';
            alpha_hat8 = squeeze(alphaBIC(Gstar,1:Gstar,:));
            res_loop_grp_hat8(simu,:) = grpBIC(Gstar,:)';
            [var_beta_hat8,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat8(simu,:)',alpha_hat8,...
                  res_loop_grp_hat8(simu,:)');
            res_G_hat8(simu,idxt) = Gstar; 
            % Compute interactive fixed effects estimators 
            % (random init, bias corrected)
            XX = zeros(K,N,T);
            for k=1:K
                XX(k,:,:) = X(:,:,k);
            end
            stop = false;
            fail = 0;
            while ~stop
                try
                    [res_loop_beta_hat9(simu,:),~,~,~,~,var_beta_hat9,~,...
                        B1,B2,B3] = LS_factor(Y,XX,G,'silent', 1e-8,'m2',...
                        zeros(K,1),n_init_IFE,n_init_IFE);
                    stop = true;
                catch
                    fail = fail + 1;
                    disp(string(fail)+...
               ' failures IFE random init, 500 it. (simu, G, N, T) =  ('...
                          +string(simu)+','+string(G)+','+string(N)+','...
                          +string(T)+').');
                end
            end
            res_loop_beta_hat10(simu,:) = res_loop_beta_hat9(simu,:)'...
                -B1-B2-B3;
            var_beta_hat10 = var_beta_hat9;
            % Compute interactive fixed effects estimators 
            %(init at beta_nnr, bias corrected)
            stop = false;
            fail = 0;
            while ~stop
                try
                    [res_loop_beta_hat11(simu,:),~,~,~,~,var_beta_hat11,...
                        ~,B1,B2,B3] = LS_factor(Y,XX,G,'silent',1e-8,...
                        'm1',res_loop_beta_hat3(simu,:)',1,1);
                    stop = true;
                 catch
                    fail = fail + 1;
                    disp(string(fail)+...
                        ' failures NNR random init (simu, G, N, T) =  ('...
                        +string(simu)+','+string(G)+','+string(N)+','...
                        +string(T)+').');
                end
            end
            res_loop_beta_hat12(simu,:) = res_loop_beta_hat11(simu,:)'...
                -B1-B2-B3;
            var_beta_hat12 = var_beta_hat11;
            % Compute oracle pooled OLS estimator
            [res_loop_beta_hat13(simu,:), alpha_hat13] = ...
                  FE_reg_withcov(Y,X,grp);
            [var_beta_hat13,~] = compute_GFE_analytical_cov(Y,X,...
                  res_loop_beta_hat13(simu,:)',alpha_hat13,grp);
            res_loop_grp_hat13(simu,:) = grp;
            
            % handle arrays of changing size in parfor loop
            alpha_hat1_pad = zeros(180,40);
            alpha_hat2_pad = zeros(180,40);
            alpha_hat6_pad = zeros(180,40);
            alpha_hat7_pad = zeros(180,40);
            alpha_hat8_pad = zeros(180,40);
            alpha_hat13_pad = zeros(180,40);

            alpha_hat1_pad(1:res_G_hat1(simu,idxt),1:T) = alpha_hat1;
            alpha_hat2_pad(1:res_G_hat2(simu,idxt),1:T) = alpha_hat2;
            alpha_hat6_pad(1:res_G_hat6(simu,idxt),1:T) = alpha_hat6;
            alpha_hat7_pad(1:G,1:T) = alpha_hat7;
            alpha_hat8_pad(1:res_G_hat8(simu,idxt),1:T) = alpha_hat8;
            alpha_hat13_pad(1:G,1:T) = alpha_hat13;

            res_loop_alpha_hat1(simu,:,:) = alpha_hat1_pad;
            res_loop_alpha_hat2(simu,:,:) = alpha_hat2_pad;
            res_loop_alpha_hat6(simu,:,:) = alpha_hat6_pad;
            res_loop_alpha_hat7(simu,:,:) = alpha_hat7_pad;
            res_loop_alpha_hat8(simu,:,:) = alpha_hat8_pad;
            res_loop_alpha_hat13(simu,:,:) = alpha_hat13_pad;
            
            res_loop_var_beta_hat1(simu,:,:) = var_beta_hat1;
            res_loop_var_beta_hat2(simu,:,:) = var_beta_hat2;
            res_loop_var_beta_hat6(simu,:,:) = var_beta_hat6;
            res_loop_var_beta_hat7(simu,:,:) = var_beta_hat7;
            res_loop_var_beta_hat8(simu,:,:) = var_beta_hat8;
            res_loop_var_beta_hat9(simu,:,:) = var_beta_hat9;
            res_loop_var_beta_hat10(simu,:,:) = var_beta_hat10;
            res_loop_var_beta_hat11(simu,:,:) = var_beta_hat11;
            res_loop_var_beta_hat12(simu,:,:) = var_beta_hat12;
            res_loop_var_beta_hat13(simu,:,:) = var_beta_hat13;


        end

        % store results and fill in table
        for simu=1:nsim
            res_grp_hat1(simu,idxt,1:N) = res_loop_grp_hat1(simu,:);
            res_grp_hat2(simu,idxt,1:N) = res_loop_grp_hat2(simu,:);
            res_grp_hat6(simu,idxt,1:N) = res_loop_grp_hat6(simu,:);
            res_grp_hat7(simu,idxt,1:N) = res_loop_grp_hat7(simu,:);
            res_grp_hat8(simu,idxt,1:N) = res_loop_grp_hat8(simu,:);
            res_grp_hat13(simu,idxt,1:N) = res_loop_grp_hat13(simu,:);
            
            res_beta_hat1(simu,idxt,:) = res_loop_beta_hat1(simu,:);
            res_beta_hat2(simu,idxt,:) = res_loop_beta_hat2(simu,:);
            res_beta_hat3(simu,idxt,:) = res_loop_beta_hat3(simu,:);
            res_beta_hat4(simu,idxt,:) = res_loop_beta_hat4(simu,:);
            res_beta_hat5(simu,idxt,:) = res_loop_beta_hat5(simu,:);
            res_beta_hat6(simu,idxt,:) = res_loop_beta_hat6(simu,:);
            res_beta_hat7(simu,idxt,:) = res_loop_beta_hat7(simu,:);
            res_beta_hat8(simu,idxt,:) = res_loop_beta_hat8(simu,:);
            res_beta_hat9(simu,idxt,:) = res_loop_beta_hat9(simu,:);
            res_beta_hat10(simu,idxt,:) = res_loop_beta_hat10(simu,:);
            res_beta_hat11(simu,idxt,:) = res_loop_beta_hat11(simu,:);
            res_beta_hat12(simu,idxt,:) = res_loop_beta_hat12(simu,:);
            res_beta_hat13(simu,idxt,:) = res_loop_beta_hat13(simu,:);
            
            res_var_beta_hat1(simu,idxt,:,:) = ...
                res_loop_var_beta_hat1(simu,:,:);
            res_var_beta_hat2(simu,idxt,:,:) = ...
                res_loop_var_beta_hat2(simu,:,:);
            res_var_beta_hat6(simu,idxt,:,:) = ...
                res_loop_var_beta_hat6(simu,:,:);
            res_var_beta_hat7(simu,idxt,:,:) = ...
                res_loop_var_beta_hat7(simu,:,:);
            res_var_beta_hat8(simu,idxt,:,:) = ...
                res_loop_var_beta_hat8(simu,:,:);
            res_var_beta_hat9(simu,idxt,:,:) = ...
                res_loop_var_beta_hat9(simu,:,:);
            res_var_beta_hat10(simu,idxt,:,:) = ...
                res_loop_var_beta_hat10(simu,:,:);
            res_var_beta_hat11(simu,idxt,:,:) = ...
                res_loop_var_beta_hat11(simu,:,:);
            res_var_beta_hat12(simu,idxt,:,:) = ...
                res_loop_var_beta_hat12(simu,:,:);
            res_var_beta_hat13(simu,idxt,:,:) = ...
                res_loop_var_beta_hat13(simu,:,:);

            res_alpha_hat1(simu,idxt,:,:) = res_loop_alpha_hat1(simu,:,:);
            res_alpha_hat2(simu,idxt,:,:) = res_loop_alpha_hat2(simu,:,:);
            res_alpha_hat6(simu,idxt,:,:) = res_loop_alpha_hat6(simu,:,:);
            res_alpha_hat7(simu,idxt,:,:) = res_loop_alpha_hat7(simu,:,:);
            res_alpha_hat8(simu,idxt,:,:) = res_loop_alpha_hat8(simu,:,:);
            res_alpha_hat13(simu,idxt,:,:) = res_loop_alpha_hat13(simu,:,:);

            res_table3(simu,idxt,1) = res_beta_hat1(simu,idxt,:)-beta;
            res_table3(simu,idxt,2) = mean((res_beta_hat1(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat1(simu,idxt,1,1));
            res_table3(simu,idxt,3) = (beta>=res_beta_hat1(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat1(simu,idxt,:)+se*norminv(1-0.05/2));

            res_table3(simu,idxt,4) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat1(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat1(simu,idxt,...
                1:res_G_hat1(simu,idxt),1:T),...
                res_G_hat1(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,5) = res_G_hat1(simu,idxt);

            res_table3(simu,idxt,6) = res_beta_hat2(simu,idxt,:)-beta;
            res_table3(simu,idxt,7) = mean((res_beta_hat2(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat2(simu,idxt,1,1));
            res_table3(simu,idxt,8) = (beta>=res_beta_hat2(simu,idxt,:)...
               -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat2(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,9) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat2(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat2(simu,idxt,1:res_G_hat2(simu,...
                idxt),1:T),res_G_hat2(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,10) = res_G_hat2(simu,idxt);
            
            res_table3(simu,idxt,11) = res_beta_hat3(simu,idxt,:)-beta;
            res_table3(simu,idxt,12) = mean((res_beta_hat3(simu,idxt,:)...
                -beta).^2);
            
            res_table3(simu,idxt,13) = res_beta_hat4(simu,idxt,:)-beta;
            res_table3(simu,idxt,14) = mean((res_beta_hat4(simu,idxt,:)...
                -beta).^2);

            res_table3(simu,idxt,15) = res_beta_hat5(simu,idxt,:)-beta;
            res_table3(simu,idxt,16) = mean((res_beta_hat5(simu,idxt,:)...
                -beta).^2);
            
            res_table3(simu,idxt,17) = res_beta_hat6(simu,idxt,:)-beta;
            res_table3(simu,idxt,18) = mean((res_beta_hat6(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat6(simu,idxt,1,1));
            res_table3(simu,idxt,19) = (beta>=res_beta_hat6(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat6(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,20) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat6(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat6(simu,idxt,1:res_G_hat6(simu,...
                idxt),1:T),res_G_hat6(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,21) = res_G_hat6(simu,idxt);
            
            res_table3(simu,idxt,22) = res_beta_hat7(simu,idxt,:)-beta;
            res_table3(simu,idxt,23) = mean((res_beta_hat7(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat7(simu,idxt,1,1));
            res_table3(simu,idxt,24) = (beta>=res_beta_hat7(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat7(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,25) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat7(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat7(simu,idxt,1:G,1:T),G,T)).^2,'all'));
            
            res_table3(simu,idxt,26) = res_beta_hat8(simu,idxt,:)-beta;
            res_table3(simu,idxt,27) = mean((res_beta_hat8(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat8(simu,idxt,1,1));
            res_table3(simu,idxt,28) = (beta>=res_beta_hat8(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat8(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,29) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat8(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat8(simu,idxt,...
                1:res_G_hat8(simu,idxt),1:T),...
                res_G_hat8(simu,idxt),T)).^2,'all'));
            res_table3(simu,idxt,30) = res_G_hat8(simu,idxt);
            
            res_table3(simu,idxt,31) = res_beta_hat9(simu,idxt,:)-beta;
            res_table3(simu,idxt,32) = mean((res_beta_hat9(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat9(simu,idxt,1,1));
            res_table3(simu,idxt,33) = (beta>=res_beta_hat9(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat9(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,34) = res_beta_hat10(simu,idxt,:)-beta;
            res_table3(simu,idxt,35) = mean((res_beta_hat10(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat10(simu,idxt,1,1));
            res_table3(simu,idxt,36) = (beta>=res_beta_hat10(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat10(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,37) = res_beta_hat11(simu,idxt,:)-beta;
            res_table3(simu,idxt,38) = mean((res_beta_hat11(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat11(simu,idxt,1,1));
            res_table3(simu,idxt,39) = (beta>=res_beta_hat11(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat11(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,40) = res_beta_hat12(simu,idxt,:)-beta;
            res_table3(simu,idxt,41) = mean((res_beta_hat12(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat12(simu,idxt,1,1));
            res_table3(simu,idxt,42) = (beta>=res_beta_hat12(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat12(simu,idxt,:)+se*norminv(1-0.05/2));
            
            res_table3(simu,idxt,43) = res_beta_hat13(simu,idxt,:)-beta;
            res_table3(simu,idxt,44) = mean((res_beta_hat13(simu,idxt,:)...
                -beta).^2);
            se = sqrt(res_var_beta_hat13(simu,idxt,1,1));
            res_table3(simu,idxt,45) = (beta>=res_beta_hat13(simu,idxt,:)...
                -se*norminv(1-0.05/2))*...
                (beta<=res_beta_hat13(simu,idxt,:)+se*norminv(1-0.05/2));
            res_table3(simu,idxt,46) = sqrt(mean((grp_dum_mat*alpha- ...
                dummyvar(reshape(res_grp_hat13(simu,idxt,1:N),1,N))*...
                reshape(res_alpha_hat13(simu,idxt,1:G,1:T),...
                G,T)).^2,'all'));
            
            res_table4(simu,idxt,1:3) = clustering_accuracy(grp, ...
                squeeze(res_grp_hat1(simu,idxt,1:N)));
            res_table4(simu,idxt,4:6) = clustering_accuracy(grp,...
                squeeze(res_grp_hat2(simu,idxt,1:N)));
            res_table4(simu,idxt,7:9) = clustering_accuracy(grp,...
                squeeze(res_grp_hat6(simu,idxt,1:N)));
            res_table4(simu,idxt,10:12) = clustering_accuracy(grp,...
                squeeze(res_grp_hat7(simu,idxt,1:N)));       
            res_table4(simu,idxt,13:15) = clustering_accuracy(grp,...
                squeeze(res_grp_hat8(simu,idxt,1:N)));
        end       
    end
    
    % Save raw results
    save(append(path,'_res_G_hat1_G',string(G),'.mat'),'res_G_hat1');
    save(append(path,'_res_G_hat2_G',string(G),'.mat'),'res_G_hat2');
    save(append(path,'_res_G_hat6_G',string(G),'.mat'),'res_G_hat6');
    save(append(path,'_res_G_hat8_G',string(G),'.mat'),'res_G_hat8');

    save(append(path,'_res_grp_hat1_G',string(G),'.mat'),'res_grp_hat1');
    save(append(path,'_res_grp_hat2_G',string(G),'.mat'),'res_grp_hat2');
    save(append(path,'_res_grp_hat6_G',string(G),'.mat'),'res_grp_hat6');
    save(append(path,'_res_grp_hat7_G',string(G),'.mat'),'res_grp_hat7');
    save(append(path,'_res_grp_hat8_G',string(G),'.mat'),'res_grp_hat8');
    save(append(path,'_res_grp_hat13_G',string(G),'.mat'),'res_grp_hat13');

    save(append(path,'_res_beta_hat1_G',string(G),'.mat'),'res_beta_hat1');
    save(append(path,'_res_beta_hat2_G',string(G),'.mat'),'res_beta_hat2');
    save(append(path,'_res_beta_hat3_G',string(G),'.mat'),'res_beta_hat3');
    save(append(path,'_res_beta_hat4_G',string(G),'.mat'),'res_beta_hat4');
    save(append(path,'_res_beta_hat5_G',string(G),'.mat'),'res_beta_hat5');
    save(append(path,'_res_beta_hat6_G',string(G),'.mat'),'res_beta_hat6');
    save(append(path,'_res_beta_hat7_G',string(G),'.mat'),'res_beta_hat7');
    save(append(path,'_res_beta_hat8_G',string(G),'.mat'),'res_beta_hat8');
    save(append(path,'_res_beta_hat9_G',string(G),'.mat'),'res_beta_hat9');
    save(append(path,'_res_beta_hat10_G',string(G),'.mat'),'res_beta_hat10');
    save(append(path,'_res_beta_hat11_G',string(G),'.mat'),'res_beta_hat11');
    save(append(path,'_res_beta_hat12_G',string(G),'.mat'),'res_beta_hat12');
    save(append(path,'_res_beta_hat13_G',string(G),'.mat'),'res_beta_hat13');
    
    save(append(path,'_res_var_beta_hat1_G',string(G),'.mat'),'res_var_beta_hat1');
    save(append(path,'_res_var_beta_hat2_G',string(G),'.mat'),'res_var_beta_hat2');
    save(append(path,'_res_var_beta_hat6_G',string(G),'.mat'),'res_var_beta_hat6');
    save(append(path,'_res_var_beta_hat7_G',string(G),'.mat'),'res_var_beta_hat7');
    save(append(path,'_res_var_beta_hat8_G',string(G),'.mat'),'res_var_beta_hat8');
    save(append(path,'_res_var_beta_hat9_G',string(G),'.mat'),'res_var_beta_hat9');
    save(append(path,'_res_var_beta_hat10_G',string(G),'.mat'),'res_var_beta_hat10');
    save(append(path,'_res_var_beta_hat11_G',string(G),'.mat'),'res_var_beta_hat11');
    save(append(path,'_res_var_beta_hat12_G',string(G),'.mat'),'res_var_beta_hat12');
    save(append(path,'_res_var_beta_hat13_G',string(G),'.mat'),'res_var_beta_hat13');

    save(append(path,'_res_alpha_hat1_G',string(G),'.mat'),'res_alpha_hat1');
    save(append(path,'_res_alpha_hat2_G',string(G),'.mat'),'res_alpha_hat2');
    save(append(path,'_res_alpha_hat6_G',string(G),'.mat'),'res_alpha_hat6');
    save(append(path,'_res_alpha_hat7_G',string(G),'.mat'),'res_alpha_hat7');
    save(append(path,'_res_alpha_hat8_G',string(G),'.mat'),'res_alpha_hat8');
    save(append(path,'_res_alpha_hat13_G',string(G),'.mat'),'res_alpha_hat13');

    
    % Save raw Table 3 (Full GFE: Bias, RMSE, Coverage, and Ghat)
    save(append(path,'_res_table3_G',string(G),'.mat'),'res_table3');

    % Output raw Table 4 (Full GFE: classification accuracy)
    save(append(path,'_res_table4_G',string(G),'.mat'),'res_table4');
   if G==3
       res_full_tab3(:,1:8,:) = res_table3;
       res_full_tab4(:,1:8,:) = res_table4;
   else
       res_full_tab3(:,9:16,:) = res_table3;
       res_full_tab4(:,9:16,:) = res_table4;
   end
end

tab3_out = reshape(mean(res_full_tab3,1),16,46);
tab3_out(:,[2,7,12,14,16,18,23,27,32,35,38,41,44]) = sqrt(tab3_out(:,...
    [2,7,12,14,16,18,23,27,32,35,38,41,44])); %RMSE beta
tab3_out = round(tab3_out,3);
save(append(path,'_table3.mat'),'tab3_out');

tab4_out = round(reshape(mean(res_full_tab4,1),16,15),3);
save(append(path,'_table4.mat'),'tab4_out');

% Output LaTex Table 3 (Part 1/2)
fileID = fopen('output_table3A.tex','w');
if SM_tables
    fileID = fopen('output_table3A_het_serr_cor.tex','w');
end
fprintf(fileID, '\\begin{tabular}{lll *{31}{S[table-format=-1.3]}}\n');
fprintf(fileID, '\\toprule\n');
fprintf(fileID, '{} & {} & {} & \\multicolumn{5}{c}{TPWD} & & \\multicolumn{5}{c}{Iterated TPWD} & & \\multicolumn{2}{c}{NNR} & &  \\multicolumn{2}{c}{NN} & &  \\multicolumn{2}{c}{Spectral} &&  \\multicolumn{5}{c}{Post-Spectral} & & \\multicolumn{4}{c}{GFE}  \\\\\n');
fprintf(fileID, '\\cmidrule{4-8}\\cmidrule{10-14}\\cmidrule{16-17}\\cmidrule{19-20}\\cmidrule{22-23}\\cmidrule{25-29}\\cmidrule{31-34} \n');
fprintf(fileID, '$G$ & $N$ & $T$ & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$}  & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$}   & \\multicolumn{1}{c}{RMSE $\\widehat\\alpha$} & \\multicolumn{1}{c}{$\\widehat G$} & & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\alpha$} & \\multicolumn{1}{c}{$\\widehat G$} & & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & &  \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & &  \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & &  \\multicolumn{1}{c}{Bias $\\widehat\\beta$}& \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$}  & \\multicolumn{1}{c}{RMSE $\\widehat\\alpha$} & \\multicolumn{1}{c}{$\\widehat G$} & & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\alpha$}  \\\\ \n'); 
fprintf(fileID, '\\midrule \n');
i = 1;
for G=[3,4]
    for N=[90,180]
        for T=Tseq
            if T==7 
                fprintf(fileID, string(G)+'&'+string(N)+'&'+string(T)+'& %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & & %.3f & %.3f & & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f  \\\\ \n', tab3_out(i,1:25));
            else
                fprintf(fileID,'& &'+string(T)+'& %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & & %.3f & %.3f & & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f  \\\\ \n', tab3_out(i,1:25));
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

% Output LaTex Table 3 (Part 2/2)
fileID = fopen('output_table3B.tex','w');
if SM_tables
    fileID = fopen('output_table3B_het_serr_cor.tex','w');
end
fprintf(fileID, '\\begin{tabular}{lll *{26}{S[table-format=-1.3]}} \n');
fprintf(fileID, '\\toprule\n');
fprintf(fileID, '{} & {} & {} & \\multicolumn{5}{c}{GFE with BIC selection} & & \\multicolumn{3}{c}{IFE R} && \\multicolumn{3}{c}{IFE R-BC} && \\multicolumn{3}{c}{IFE NNR} && \\multicolumn{3}{c}{IFE NNR-BC}& &  \\multicolumn{4}{c}{Oracle}  \\\\\n');
fprintf(fileID, '\\cmidrule{4-8}\\cmidrule{10-12}\\cmidrule{14-16}\\cmidrule{18-20}\\cmidrule{22-24}\\cmidrule{26-29} \n');
fprintf(fileID, '$G$ & $N$ & $T$ & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\alpha$} & \\multicolumn{1}{c}{$\\widehat G$} & & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$} & &  \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$} & & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$}  & & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$} & & \\multicolumn{1}{c}{Bias $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\beta$} & \\multicolumn{1}{c}{$.95$ $\\widehat\\beta$} & \\multicolumn{1}{c}{RMSE $\\widehat\\alpha$}  \\\\ \n');
fprintf(fileID, '\\midrule \n');
i = 1;
for G=[3,4]
    for N=[90,180]
        for T=Tseq
            if T==7 
                fprintf(fileID, string(G)+'&'+string(N)+'&'+string(T)+'& %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f  \\\\ \n', tab3_out(i,26:end));
            else
                fprintf(fileID,'& &'+string(T)+'& %.3f & %.3f & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & %.3f  \\\\ \n', tab3_out(i,26:end));
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

% Output LaTex Tables 4
fileID = fopen('output_table4.tex','w');
if SM_tables
    fileID = fopen('output_table4_het_serr_cor.tex','w');
end
fprintf(fileID, '\\begin{tabular}{lll *{19}{S[table-format=-1.3]}} \n');
fprintf(fileID, '\\toprule\n');
fprintf(fileID, '{} & {} & {} & \\multicolumn{3}{c}{TPWD} & & \\multicolumn{3}{c}{Iterated TPWD} & & \\multicolumn{3}{c}{Post-Spectral} & & \\multicolumn{3}{c}{GFE} & & \\multicolumn{3}{c}{GFE with BIC selection} \\\\\n');
fprintf(fileID, '\\cmidrule{4-6}\\cmidrule{8-10}\\cmidrule{12-14}\\cmidrule{16-18}\\cmidrule{20-22}\n');
fprintf(fileID, '$G$ &$N$ & $T$ &  P & R & RI && P & R & RI && P & R & RI && P & R & RI && P & R & RI  \\\\\n');
fprintf(fileID, '\\midrule \n');
i = 1;
for G=[3,4]
    for N=[90,180]
        for T=Tseq
            if T==7 
                fprintf(fileID, string(G)+'&'+string(N)+'&'+string(T)+'& %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f   \\\\ \n', tab4_out(i,:));
            else
                fprintf(fileID,'& &'+string(T)+'& %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f & & %.3f & %.3f & %.3f   \\\\ \n', tab4_out(i,:));
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