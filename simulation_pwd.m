% Reproducibility
rng('default');

% Monte Carlo parameters
G_seq = [2 5 10 50];
N_seq = [50 100 200 500];
B = 1000; % number of simulated samples
sigma = 1; % noise

res = zeros(length(N_seq),4,length(G_seq),6,B); % N times T times G times n_res times B
c = 2; % threshold

% Run simulations
% DGP 1
%noise = normrnd(0, sigma, B, 500, 500);
% DGP 2
%noise_mdl = arima('Constant',0,'AR',{0.5},'Variance',sigma^2);
%noise = zeros(B,500,500);
%for b=1:B
%    noise(b,:,:) = simulate(noise_mdl, 500,'NumPaths',500)';
%end
% DGP 3
Sigma = diag((0.5 + rand(500, 1)).^2);
noise = zeros(B,500,500);
for b=1:B
    noise(b,:,:) = mvnrnd(zeros(500, 1),Sigma,500);
end

g = 1
for G=G_seq
    n_idx = 1
    for n=N_seq
        true_grp = repelem([1:G],n/G)';
        dist_true = abs(bsxfun(@minus,true_grp,true_grp'))>0;
        true_grp_effects = linspace(-G/2,G/2,G);
        true_signal = repelem(true_grp_effects,n/G);
        t = 1;
        for T=ceil(linspace(sqrt(n), n, 4))
            for b=1:B
                Y = reshape(repelem(true_signal, T),T,n)' + reshape(noise(b,1:n,1:T), n, T);
                start = cputime;
                [G_hat,grp,grp_effects] = pwd_estimators.pwd1(Y,c*log(T)/sqrt(T));
                CPU = cputime-start;
                HD = HausdorffDist(grp_effects, true_grp_effects');
                dist_hat = abs(bsxfun(@minus,grp,grp'))>0;
                FP = sum((1-dist_hat).*dist_true, 'all')/2;
                TP = (sum((1-dist_hat).*(1-dist_true), 'all')-n)/2;
                FN = sum(dist_hat.*(1-dist_true), 'all')/2;
                TN = sum(dist_hat.*dist_true, 'all')/2;
                P = TP/(TP+FP);
                R = TP/(FN+TP);
                RI =(TP+TN)/(TP+FP+FN+TN);
                res(n_idx, t, g, :, b) = [G_hat, HD, RI, P, R, CPU];
            end
            t = t + 1
        end
        n_idx = n_idx +1
    end
    g = g + 1
end

export = mean(res, 5);
export_bis = zeros(16, 16);
export_bis(:,1) = cat(1, reshape(export(1,:,1,1), 4, 1), reshape(export(2,:,1,1), 4, 1), reshape(export(3,:,1,1), 4, 1), reshape(export(4,:,1,1), 4, 1)); 
export_bis(:,2) = cat(1, reshape(export(1,:,1,2), 4, 1), reshape(export(2,:,1,2), 4, 1), reshape(export(3,:,1,2), 4, 1), reshape(export(4,:,1,2), 4, 1));
export_bis(:,3) = cat(1, reshape(export(1,:,1,3), 4, 1), reshape(export(2,:,1,3), 4, 1), reshape(export(3,:,1,3), 4, 1), reshape(export(4,:,1,3), 4, 1)); 
export_bis(:,4) = cat(1, reshape(export(1,:,1,6), 4, 1), reshape(export(2,:,1,6), 4, 1), reshape(export(3,:,1,6), 4, 1), reshape(export(4,:,1,6), 4, 1)); 
export_bis(:,5) = cat(1, reshape(export(1,:,2,1), 4, 1), reshape(export(2,:,2,1), 4, 1), reshape(export(3,:,2,1), 4, 1), reshape(export(4,:,2,1), 4, 1)); 
export_bis(:,6) = cat(1, reshape(export(1,:,2,2), 4, 1), reshape(export(2,:,2,2), 4, 1), reshape(export(3,:,2,2), 4, 1), reshape(export(4,:,2,2), 4, 1));
export_bis(:,7) = cat(1, reshape(export(1,:,2,3), 4, 1), reshape(export(2,:,2,3), 4, 1), reshape(export(3,:,2,3), 4, 1), reshape(export(4,:,2,3), 4, 1)); 
export_bis(:,8) = cat(1, reshape(export(1,:,2,6), 4, 1), reshape(export(2,:,2,6), 4, 1), reshape(export(3,:,2,6), 4, 1), reshape(export(4,:,2,6), 4, 1)); 
export_bis(:,9) = cat(1, reshape(export(1,:,3,1), 4, 1), reshape(export(2,:,3,1), 4, 1), reshape(export(3,:,3,1), 4, 1), reshape(export(4,:,3,1), 4, 1)); 
export_bis(:,10) = cat(1, reshape(export(1,:,3,2), 4, 1), reshape(export(2,:,3,2), 4, 1), reshape(export(3,:,3,2), 4, 1), reshape(export(4,:,3,2), 4, 1));
export_bis(:,11) = cat(1, reshape(export(1,:,3,3), 4, 1), reshape(export(2,:,3,3), 4, 1), reshape(export(3,:,3,3), 4, 1), reshape(export(4,:,3,3), 4, 1)); 
export_bis(:,12) = cat(1, reshape(export(1,:,3,6), 4, 1), reshape(export(2,:,3,6), 4, 1), reshape(export(3,:,3,6), 4, 1), reshape(export(4,:,3,6), 4, 1)); 
export_bis(:,13) = cat(1, reshape(export(1,:,4,1), 4, 1), reshape(export(2,:,4,1), 4, 1), reshape(export(3,:,4,1), 4, 1), reshape(export(4,:,4,1), 4, 1)); 
export_bis(:,14) = cat(1, reshape(export(1,:,4,2), 4, 1), reshape(export(2,:,4,2), 4, 1), reshape(export(3,:,4,2), 4, 1), reshape(export(4,:,4,2), 4, 1));
export_bis(:,15) = cat(1, reshape(export(1,:,4,3), 4, 1), reshape(export(2,:,4,3), 4, 1), reshape(export(3,:,4,3), 4, 1), reshape(export(4,:,4,3), 4, 1)); 
export_bis(:,16) = cat(1, reshape(export(1,:,4,6), 4, 1), reshape(export(2,:,4,6), 4, 1), reshape(export(3,:,4,6), 4, 1), reshape(export(4,:,4,6), 4, 1)); 

save('save_results_DGP3.mat','export_bis');
latex_table = latex(sym(vpa(round(export_bis, 4))));
save('results_DGP3.txt','latex_table');