%--------------------- MONTE CARLO SIMULATIONS ----------------------------
% This code produces Tables:
%  - 1,2: iid errors, signal-to-noise ratio=1;
%  - S1,S2: iid errors, signal-to-noise ratio=1, unbalanced groups;
%  - S3,S4: dependent errors, signal-to-noise ratio=1;
%  - S5,S6: iid errors, signal-to-noise ratio=1/2;
%--------------------------------------------------------------------------

clear;
rng('default'); 

G = 3; 
%G = 4;  %Unmute if G=4
Tseq = [7,10,20,40];
sigma = 1/3; %Change to 2/3 to output Tables S5 and S6

%Simulations
nsim = 500;
res = [];

G_hat_res = [];
grp_hat_1res = [];
grp_hat_2res = [];
grp_hat_3res = [];
grp_hat_4res = [];
alphahat_1res = [];
alphahat_2res = [];
alphahat_3res = [];
alphahat_4res = [];

%Panel A 
N = 90;
grp = repelem([1:G],N/G)'; %Unmute if 3 balanced groups
%grp = 1+((1:N)>22)+((1:N)>44)+((1:N)>66); %Unmute if 4 balanced groups
%grp = 1+((1:N)>2)+((1:N)>N/10); %Unmute if 3 unbalanced groups (S1, S2)
%grp = 1+((1:N)>2)+((1:N)>N/10)+((1:N)>N/2); %Unmute if 4 unbalanced groups (S1, S2)
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
        %noise_mdl = arima('Constant',0,'AR',{0.20},'Variance',sigma^2); %Unmute to produce S3,S4
        %err = simulate(noise_mdl,T,'NumPaths',N)'; %Unmute to produce S3,S4
        Y = grp_dum_mat*alpha+err;
        sigmahat = std(Y,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        tic
        [Ghat1,grphat1,alphahat1] = tpwd_pureGFE(Y,c_opt,2);
        wct = toc;
        res(simu,idxt,1) = Ghat1; 
        G_hat_res(simu,idxt,1) = Ghat1;
        grp_hat_1res(simu,idxt,1:N) = grphat1;
        alphahat_1res(simu,idxt,1:Ghat1,1:T) = alphahat1;
        res(simu,idxt,2) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat1)*alphahat1).^2,'all'));
        res(simu,idxt,3:5) = clustering_accuracy(grp,grphat1);
        res(simu,idxt,6) = wct; 
        [grphat2,alphahat2] = GFE(Y,2,10000);
        grp_hat_2res(simu,idxt,1:N) = grphat2;
        alphahat_2res(simu,idxt,1:2,1:T) = alphahat2;
        res(simu,idxt,7) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat2)*alphahat2).^2,'all'));
        res(simu,idxt,8:10) = clustering_accuracy(grp,grphat2);
        [grphat3,alphahat3] = GFE(Y,3,10000);
        grp_hat_3res(simu,idxt,1:N) = grphat3;
        alphahat_3res(simu,idxt,1:3,1:T) = alphahat3;
        res(simu,idxt,11) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat3)*alphahat3).^2,'all'));
        res(simu,idxt,12:14) = clustering_accuracy(grp,grphat3); 
        [grphat4,alphahat4] = GFE(Y,10,10000);
        grp_hat_4res(simu,idxt,1:N) = grphat4;
        alphahat_4res(simu,idxt,1:10,1:T) = alphahat4;
        res(simu,idxt,15) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat4)*alphahat4).^2,'all'));
        res(simu,idxt,16:18) = clustering_accuracy(grp,grphat4); 
    end
end

%Panel B 
N = 180;
grp = repelem([1:G],N/G)'; %Unmute if 3 balanced groups
%grp = 1+((1:N)>22)+((1:N)>44)+((1:N)>66); %Unmute if 4 balanced groups
%grp = 1+((1:N)>2)+((1:N)>N/10); %Unmute if 3 unbalanced groups (S1, S2)
%grp = 1+((1:N)>2)+((1:N)>N/10)+((1:N)>N/2); %Unmute if 4 unbalanced groups (S1, S2)
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
        %noise_mdl = arima('Constant',0,'AR',{0.20},'Variance',sigma^2); %Unmute to produce S3,S4
        %err = simulate(noise_mdl,T,'NumPaths',N)'; %Unmute to produce S3,S4
        Y = grp_dum_mat*alpha+err;
        sigmahat = std(Y,1,'all'); 
        c_opt = sigmahat*log(T)*T^(-1/2);
        tic
        [Ghat1,grphat1,alphahat1] = tpwd_pureGFE(Y,c_opt,2);
        wct = toc;
        res(simu,idxt,1) = Ghat1; 
        G_hat_res(simu,idxt,1) = Ghat1;
        grp_hat_1res(simu,idxt,1:N) = grphat1;
        alphahat_1res(simu,idxt,1:Ghat1,1:T) = alphahat1;
        res(simu,idxt,2) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat1)*alphahat1).^2,'all'));
        res(simu,idxt,3:5) = clustering_accuracy(grp,grphat1);
        res(simu,idxt,6) = wct; 
        [grphat2,alphahat2] = GFE(Y,2,10000);
        grp_hat_2res(simu,idxt,1:N) = grphat2;
        alphahat_2res(simu,idxt,1:2,1:T) = alphahat2;
        res(simu,idxt,7) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat2)*alphahat2).^2,'all'));
        res(simu,idxt,8:10) = clustering_accuracy(grp,grphat2);
        [grphat3,alphahat3] = GFE(Y,3,10000);
        grp_hat_3res(simu,idxt,1:N) = grphat3;
        alphahat_3res(simu,idxt,1:3,1:T) = alphahat3;
        res(simu,idxt,11) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat3)*alphahat3).^2,'all'));
        res(simu,idxt,12:14) = clustering_accuracy(grp,grphat3); 
        [grphat4,alphahat4] = GFE(Y,10,10000);
        grp_hat_4res(simu,idxt,1:N) = grphat4;
        alphahat_4res(simu,idxt,1:10,1:T) = alphahat4;
        res(simu,idxt,15) = sqrt(mean((grp_dum_mat*alpha- ...
            dummyvar(grphat4)*alphahat4).^2,'all'));
        res(simu,idxt,16:18) = clustering_accuracy(grp,grphat4); 
    end
end

save('mc_res_pure_gfe_G3.mat','res')
save('mc_G_hat_res_pure_gfe_G3.mat','G_hat_res')
save('mc_grp_hat_1res_pure_gfe_G3.mat','grp_hat_1res')
save('mc_alphahat_1res_pure_gfe_G3.mat','alphahat_1res')
save('mc_grp_hat_2res_pure_gfe_G3.mat','grp_hat_2res')
save('mc_alphahat_2res_pure_gfe_G3.mat','alphahat_2res')
save('mc_grp_hat_3res_pure_gfe_G3.mat','grp_hat_3res')
save('mc_alphahat_3res_pure_gfe_G3.mat','alphahat_3res')
save('mc_grp_hat_4res_pure_gfe_G3.mat','grp_hat_4res')
save('mc_alphahat_4res_pure_gfe_G3.mat','alphahat_4res')

% Table 1 (Pure GFE: Ghat, RMSE and CPU time)
tab1_out = round(reshape(mean(res(:,1:8,[1 2 6 7 11 15])),8,6),3);
disp(tab1_out);
save('Table1_G3.mat','tab1_out')

% Table 2 (Pure GFE: classification accuracy)
tab2_out = round(reshape(mean(res(:,1:8,[3 4 5 8 9 10 12 13 14 16 17 18])),8,12),3);
disp(tab2_out);
save('Table2_G3.mat','tab2_out')


