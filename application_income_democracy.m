%----------------- APPLICATION: INCOME AND DEMOCRACY ----------------------
% This code produces Table 5.
%--------------------------------------------------------------------------

clear;

%Load and clean data
M = csvread('acemoglu_balancedsample.csv');

N = 90; T=7; K=2;
Y = reshape(M(:,1),T,N)';
X(:,:,1) = reshape(M(:,2),T,N)';
X(:,:,2) = reshape(M(:,3),T,N)';

%Plot data
histogram(Y,20);

%Compute NNR preliminary estimate
beta0 = MW_nucnormreg_estimator_optim(Y,X,log(log(T))/sqrt(16*T));

link = 'average';

n_init_GFE = 100;

% Compute GFE estimator (BIC crit. to select groups with Gmax=5)
%Gmax = N/2;
Gmax=5;
objBIC = zeros(Gmax,1);
betaBIC = zeros(Gmax,K);
alphaBIC = zeros(Gmax,Gmax,T);
grpBIC = zeros(Gmax,N);
[betaBIC(Gmax,:),alphaBIC(Gmax,1:Gmax,:),grpBIC(Gmax,:),objBIC(Gmax)] ...
    = BM_algo1_multiple_init(Y,X,Gmax,n_init_GFE);
G_BIC_list = 1:Gmax-1;
for j=G_BIC_list
    [betaBIC(j,:),alphaBIC(j,1:j,:),grpBIC(j,:),objBIC(j)] = ...
        BM_algo1_multiple_init(Y,X,j,n_init_GFE);
    disp(j)
end
sig_sq_BIC = objBIC(Gmax)/(N*T-Gmax*T-N-K);
objBIC = objBIC/(N*T) +reshape(sig_sq_BIC*((1:Gmax)*T+N+K)/(N*T)*log(N*T),...
    Gmax,1);
[~,Gstar] = min(objBIC);
beta_GFE_selec = betaBIC(Gstar,:)';
alpha_GFE_selec = reshape(alphaBIC(Gstar,1:Gstar,:), Gstar,T);
grp_GFE_selec = grpBIC(Gstar,:)';
[var_beta_GFE_selec,~] = compute_GFE_analytical_cov(Y,X,beta_GFE_selec,...
    alpha_GFE_selec,grp_GFE_selec);
G_GFE_selec = Gstar; 
plot(alpha_GFE_selec')
          
%TPWDit1 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta0',ones([90,7])),90,7,2),3);
sigma1 = std(residual,0,'all');
c1 = 1.5*sigma1^2*log(7)/sqrt(7);
[Ghat1,grphat1]  = TPWD_clustering(residual,c1,link);
%Obtain final estimates
[beta1,alphahat1] =  FE_reg_withcov(Y,X,grphat1);
%Regularization path
n_c_grid = 1000;
c_grid = linspace(0.01,0.14, n_c_grid); 
model_selec1 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec1(c) = Ghat;
end
plot(c_grid, model_selec1);
xline(c1);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd1.eps');

%TPWDit2 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta1',ones([90,7])),90,7,2),3);
sigma2 = std(residual,0,'all');
c2 = 1.5*sigma2^2*log(7)/sqrt(7); 
[Ghat2,grphat2]  = TPWD_clustering(residual,c2,link);
%Obtain final estimates
[beta2,alphahat2] =  FE_reg_withcov(Y,X,grphat2);
plot(alphahat2')
%Regularization path
model_selec2 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec2(c) = Ghat;
end
plot(c_grid, model_selec2);
xline(c2);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd2.eps');

%TPWDit3 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta2',ones([90,7])),90,7,2), 3);
sigma3 = std(residual,0,'all');
c3 = 1.5*sigma3^2*log(7)/sqrt(7);
[Ghat3,grphat3]  = TPWD_clustering(residual,c3,link);
%Obtain final estimates
[beta3,alphahat3] =  FE_reg_withcov(Y,X,grphat3);
plot(alphahat3')
%Regularization path
model_selec3 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec3(c) = Ghat;
end
plot(c_grid, model_selec3);
xline(c3);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd3.eps');

%TPWDit4 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta3',ones([90,7])),90,7,2), 3);
sigma4 = std(residual,0,'all');
c4 = 1.5*sigma4^2*log(7)/sqrt(7);
[Ghat4,grphat4]  = TPWD_clustering(residual,c4,link);
%Obtain final estimates
[beta4,alphahat4] =  FE_reg_withcov(Y,X,grphat4);
plot(alphahat4')
%Regularization path
model_selec4 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec4(c) = Ghat;
end
plot(c_grid, model_selec4);
xline(c4);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd4.eps');

%TPWDit5 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta4',ones([90,7])),90,7,2), 3);
sigma5 = std(residual,0,'all');
c5 = 1.5*sigma5^2*log(7)/sqrt(7);
[Ghat5,grphat5]  = TPWD_clustering(residual,c5,link);
%Obtain final estimates
[beta5,alphahat5] =  FE_reg_withcov(Y,X,grphat5);
%Regularization path
model_selec5 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec5(c) = Ghat;
end
plot(c_grid,model_selec5);
xline(c5);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd5.eps');

%TPWDit6 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta5',ones([90,7])),90,7,2), 3);
sigma6 = std(residual,0,'all');
c6 = 1.5*sigma6^2*log(7)/sqrt(7);
[Ghat6,grphat6]  = TPWD_clustering(residual,c6,link);
%Obtain final estimates
[beta6,alphahat6] =  FE_reg_withcov(Y,X,grphat6);
plot(alphahat6')
%Regularization path
model_selec6 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec5(c) = Ghat;
end
plot(c_grid,model_selec6);
xline(c6);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd6.eps');

%TPWDit7 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta6',ones([90,7])),90,7,2), 3);
sigma7 = std(residual,0,'all');
c7 = 1.5*sigma7^2*log(7)/sqrt(7);
[Ghat7,grphat7]  = TPWD_clustering(residual,c7,link);
%Obtain final estimates
[beta7,alphahat7] =  FE_reg_withcov(Y,X,grphat7);
plot(alphahat7')
%Regularization path
model_selec7 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec7(c) = Ghat;
end
plot(c_grid, model_selec7);
xline(c7);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd7.eps');

%TPWDit8 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta7',ones([90,7])),90,7,2), 3);
sigma8 = std(residual,0,'all');
c8 = 1.5*sigma8^2*log(7)/sqrt(7);
[Ghat8,grphat8]  = TPWD_clustering(residual,c8,link);
%Obtain final estimates
[beta8,alphahat8] =  FE_reg_withcov(Y,X,grphat8);
plot(alphahat8')
%Regularization path
model_selec8 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec8(c) = Ghat;
end
plot(c_grid, model_selec8);
xline(c8);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd8.eps');

plot(linspace(1970,2000,7), alphahat8(1,:), '-',linspace(1970,2000,7), ...
    alphahat8(2,:),'--', linspace(1970,2000,7), alphahat8(3,:),'-.');
xlabel('Years','Interpreter','latex');
ylabel('Time effects','Interpreter','latex');
fig = gcf;
fig.Units               = 'centimeters';
fig.Position(3)         = 12;
fig.Position(4)         = 9;
hgexport(fig, 'application_income_dem_timeeffects_it8.eps');



%TPWDit9 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta8',ones([90,7])),90,7,2), 3);
sigma9 = std(residual,0,'all');
c9 = 1.5*sigma9^2*log(7)/sqrt(7);
[Ghat9,grphat9]  = TPWD_clustering(residual,c9,link);
%Obtain final estimates
[beta9,alphahat9] =  FE_reg_withcov(Y,X,grphat9);
plot(alphahat9')

plot(linspace(1970,2000,7), alphahat9(1,:), '-',linspace(1970,2000,7), ...
    alphahat9(2,:),'--', linspace(1970,2000,7), alphahat9(3,:),'-.');
xlabel('Years','Interpreter','latex');
ylabel('Time effects','Interpreter','latex');
fig = gcf;
fig.Units               = 'centimeters';
fig.Position(3)         = 12;
fig.Position(4)         = 9;
hgexport(fig, 'application_income_dem_timeeffects_it9.eps');


%TPWDit10 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta9',ones([90,7])),90,7,2), 3);
sigma10 = std(residual,0,'all');
c10 = 1.5*sigma10^2*log(7)/sqrt(7);
[Ghat10,grphat10]  = TPWD_clustering(residual,c10,link);
%Obtain final estimates
[beta10,alphahat10] =  FE_reg_withcov(Y,X,grphat10);
plot(alphahat10')
%Regularization path
model_selec10 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec10(c) = Ghat;
end
plot(c_grid, model_selec10);
xline(c10);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd10.eps');

%TPWDit11 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta10',ones([90,7])),90,7,2), 3);
sigma11 = std(residual,0,'all');
c11 = 1.5*sigma11^2*log(7)/sqrt(7);
[Ghat11,grphat11]  = TPWD_clustering(residual,c11,link);
%Obtain final estimates
[beta11,alphahat11] =  FE_reg_withcov(Y,X,grphat11);
plot(alphahat11')
%Regularization path
model_selec11 = [];
parfor c=1:n_c_grid
    [Ghat,~] = TPWD_clustering(residual,c_grid(c),link);
    model_selec11(c) = Ghat;
end
plot(c_grid, model_selec11);
xline(c11);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd11.eps');
%CONVERGENCE


%TPWDit12 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta11',ones([90,7])),90,7,2), 3);
sigma12 = std(residual,0,'all');
c12 = 1.5*sigma12^2*log(7)/sqrt(7);
[Ghat12,grphat12]  = TPWD_clustering(residual,c12,link);
%Obtain final estimates
[beta12,alphahat12] =  FE_reg_withcov(Y,X,grphat12);
plot(alphahat12')


%TPWDit13 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta12',ones([90,7])),90,7,2), 3);
sigma13 = std(residual,0,'all');
c13 = 1.5*sigma13^2*log(7)/sqrt(7);
[Ghat13,grphat13]  = TPWD_clustering(residual,c13,link);
%Obtain final estimates
[beta13,alphahat13] =  FE_reg_withcov(Y,X,grphat13);
plot(alphahat13')

% Make Figure 1
h1 = plot(c_grid, model_selec1, '-', 'Color', 'blue', 'DisplayName', 'Iteration 1');
xline(c1, '-', 'Color', 'blue');
hold on
h2 = plot(c_grid, model_selec2, '--', 'Color', 'green', 'DisplayName', 'Iteration 2');
xline(c2, '--', 'Color', 'green');
hold on
h3 = plot(c_grid, model_selec7, '--', 'Color', 'red', 'DisplayName', 'Iteration 7');
xline(c7, ':', 'Color', 'red');
hold on
h4 = plot(c_grid, model_selec8, '-.', 'Color', 'magenta', 'DisplayName', 'Iteration 8');
xline(c8, '-.', 'Color', 'magenta');
hold off

xlabel('Thresholding parameter', 'Interpreter', 'latex');
ylabel('Estimated number of groups', 'Interpreter', 'latex');

% Add legend
legend([h1, h2, h3, h4], 'Location', 'best');

fig = gcf;
fig.Units = 'centimeters';
fig.Position(3) = 12;
fig.Position(4) = 9;

hgexport(fig, 'application_income_dem_pwd_combo.eps');

% cumulative effects
cum0 = beta0(2)/(1-beta0(1));
cum1 = beta1(2)/(1-beta1(1));
cum2 = beta2(2)/(1-beta2(1));
cum3 = beta3(2)/(1-beta3(1));
cum4 = beta4(2)/(1-beta4(1));
cum10 = beta10(2)/(1-beta10(1));
cum11 = beta11(2)/(1-beta11(1));

% Standard Errors
[A,~] = compute_GFE_analytical_cov(Y,X,beta1,alphahat1,grphat1);
avbeta1 = [sqrt(A(1,1)) sqrt(A(2,2))];
grad = [beta1(2)/(1-beta1(1))^2 1/(1-beta1(1))];
avcum1 = sqrt(grad*A*grad');

[A,~] = compute_GFE_analytical_cov(Y,X,beta2,alphahat2,grphat2);
avbeta2 = [sqrt(A(1,1)) sqrt(A(2,2))];
grad = [beta2(2)/(1-beta2(1))^2 1/(1-beta2(1))];
avcum2 = sqrt(grad*A*grad');

[A,~] = compute_GFE_analytical_cov(Y,X,beta10,alphahat10,grphat10);
avbeta10 = [sqrt(A(1,1)) sqrt(A(2,2))];
grad = [beta10(2)/(1-beta10(1))^2 1/(1-beta10(1))];
avcum10 = sqrt(grad*A*grad');

[A,~] = compute_GFE_analytical_cov(Y,X,beta11,alphahat11,grphat11);
avbeta11 = [sqrt(A(1,1)), sqrt(A(2,2))];
grad = [beta11(2)/(1-beta11(1))^2 1/(1-beta11(1))];
avcum11 = sqrt(grad*A*grad');

% Output LaTex Table 5
fileID = fopen('output_table5.tex','w');
fprintf(fileID, '\\begin{tabular}{l *{8}{c}}\n');
fprintf(fileID, '\\toprule\n');
fprintf(fileID, '& NNR & TPWD$^{\\rm 1 it}$ & TPWD$^{\\rm 2 it}$ &  TPWD$^{\\rm 10 it}$ & TPWD$^{\\rm 11 it}$   & GFE$^{G=2}$& GFE$^{G=3}$& GFE$^{G=10}$ \\\\\n');
fprintf(fileID, 'Dependent variable: Democracy \\\\ \n');
fprintf(fileID, '\\midrule\n');
fprintf(fileID, '$\\widehat G$ & - & %.3f & %.3f & %.3f & %.3f & - & - & -  \\\\ \n', [Ghat1 Ghat2 Ghat10 Ghat11]);
fprintf(fileID,'Lagged Democracy $(\\beta_1)$ & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f  \\\\ \n', [beta0(1) beta1(1) beta2(1) beta10(1) beta11(1) 0.601 0.407 0.277]);
fprintf(fileID, '& & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f)  \\\\ \n', [avbeta1(1) avbeta2(1) avbeta10(1) avbeta11(1) 0.041  0.052  0.049]);

fprintf(fileID,'Lagged Income $(\\beta_2)$ & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f  \\\\ \n', [beta0(2) beta1(2) beta2(2) beta10(2) beta11(2) 0.061 0.089 0.075]);
fprintf(fileID, '& & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f)  \\\\ \n', [avbeta1(2) avbeta2(2) avbeta10(2) avbeta11(2) 0.011 0.011 0.008]);
fprintf(fileID,'Cumulative Income $\\left(\\frac{\\beta_2}{1-\\beta_1}\\right)$ & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f  \\\\ \n', [cum0 cum1 cum2 cum10 cum11 0.152 0.151 0.104]);
fprintf(fileID, '& & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f) & (%.3f)  \\\\ \n', [avcum1 avcum2 avcum10 avcum11 0.021 0.013 0.009]);		
fprintf(fileID, '\\bottomrule\n');
fprintf(fileID, '\\end{tabular}\n');
fclose(fileID);



