%----------------- APPLICATION: INCOME AND DEMOCRACY ----------------------
% This code produces Figures 1,2 and Table 5.
%--------------------------------------------------------------------------

clear all;

%Load and clean data
M = csvread('acemoglu_balancedsample.csv');
Y = reshape(M(:,1), 7, 90)';
X(:,:,1) = reshape(M(:,2), 7, 90)';
X(:,:,2) = reshape(M(:,3), 7, 90)';

%Plot data
histogram(Y,20);

%Compute preliminary estimate
beta0 = nucnorm_reg(Y,X,log(log(7))/sqrt(16*7));

%TPWDit1 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta0',ones([90,7])),90,7,2),3);
sigma1 = std(residual,1,'all');
c1 = sigma1*log(7)/sqrt(7);
[Ghat1,grphat1,~]  = tpwd_pureGFE(residual,c1,2);
%Obtain final estimates
[beta1,alphahat1] =  FE_reg_withcov(Y,X,grphat1,true);
plot(alphahat1')
%Regularization path
n_c_grid = 1000;
disp(c1+0.02);
c_grid = linspace(0,c1+0.02, n_c_grid); 
model_selec1 = [];
parfor c=1:n_c_grid
    [Ghat,~,~] = tpwd_pureGFE(residual,c_grid(c),2);
    model_selec1(c) = Ghat;
end
plot(c_grid, model_selec1);
xline(c1);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd1.eps');

%TPWDit2 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta1',ones([90,7])),90,7,2), 3);
sigma2 = std(residual,1,'all');
c2 = sigma2*log(7)/sqrt(7);
[Ghat2,grphat2,~]  = tpwd_pureGFE(residual,c2,2);
%Obtain final estimates
[beta2,alphahat2] =  FE_reg_withcov(Y,X,grphat2,true);
plot(alphahat2')
%Regularization path
model_selec2 = [];
parfor c=1:n_c_grid
    [Ghat,~,~] = tpwd_pureGFE(residual,c_grid(c),2);
    model_selec2(c) = Ghat;
end
plot(c_grid, model_selec2);
xline(c1);
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
hgexport(fig, 'application_income_dem_pwd2.eps');

%TPWDit3 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta2',ones([90,7])),90,7,2), 3);
sigma3 = std(residual,1,'all');
c3 = sigma3*log(7)/sqrt(7);
[Ghat3,grphat3,~]  = tpwd_pureGFE(residual,c3,2);
%Obtain final estimates
[beta3,alphahat3] =  FE_reg_withcov(Y,X,grphat3,true);
plot(alphahat3');
%Regularization path
model_selec3 = [];
parfor c=1:n_c_grid
    [Ghat,~,~] = tpwd_pureGFE(residual,c_grid(c),2);
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
sigma4 = std(residual,1,'all');
c4 = sigma4*log(7)/sqrt(7);
[Ghat4,grphat4,~]  = tpwd_pureGFE(residual,c4,2);
%Obtain final estimates
[beta4,alphahat4] =  FE_reg_withcov(Y,X,grphat4,true);
%Regularization path
model_selec4 = [];
parfor c=1:n_c_grid
    [Ghat,~,~] = tpwd_pureGFE(residual,c_grid(c),2);
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
sigma5 = std(residual,1,'all');
c5 = sigma5*log(7)/sqrt(7);
[Ghat5,grphat5,~]  = tpwd_pureGFE(residual,c5,2);
%Obtain final estimates
[beta5,alphahat5] =  FE_reg_withcov(Y,X,grphat5,true);

%TPWDit6 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta5',ones([90,7])),90,7,2), 3);
sigma6 = std(residual,1,'all');
c6 = sigma6*log(7)/sqrt(7);
[Ghat6,grphat6,~]  = tpwd_pureGFE(residual,c6,2);
%Obtain final estimates
[beta6,alphahat6] =  FE_reg_withcov(Y,X,grphat6,true);

%TPWDit7 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta6',ones([90,7])),90,7,2), 3);
sigma7 = std(residual,1,'all');
c7 = sigma7*log(7)/sqrt(7);
[Ghat7,grphat7,~]  = tpwd_pureGFE(residual,c7,2);
%Obtain final estimates
[beta7,alphahat7] =  FE_reg_withcov(Y,X,grphat7,true);

%TPWDit8 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta7',ones([90,7])),90,7,2), 3);
sigma8 = std(residual,1,'all');
c8 = sigma7*log(7)/sqrt(7);
[Ghat8,grphat8,~]  = tpwd_pureGFE(residual,c8,2);
%Obtain final estimates
[beta8,alphahat8] =  FE_reg_withcov(Y,X,grphat8,true);
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
sigma9 = std(residual,1,'all');
c9 = sigma9*log(7)/sqrt(7);
[Ghat9,grphat9,~]  = tpwd_pureGFE(residual,c9,2);
%Obtain final estimates
[beta9,alphahat9] =  FE_reg_withcov(Y,X,grphat9,true);
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
sigma10 = std(residual,1,'all');
c10 = sigma10*log(7)/sqrt(7);
[Ghat10,grphat10,~]  = tpwd_pureGFE(residual,c10,2);
%Obtain final estimates
[beta10,alphahat10] =  FE_reg_withcov(Y,X,grphat10,true);

%TPWDit11 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta10',ones([90,7])),90,7,2), 3);
sigma11 = std(residual,1,'all');
c11 = sigma11*log(7)/sqrt(7);
[Ghat11,grphat11,~]  = tpwd_pureGFE(residual,c11,2);
%Obtain final estimates
[beta11,alphahat11] =  FE_reg_withcov(Y,X,grphat11,true);

%TPWDit12 with data-driven bandwidth
residual = Y-sum(X.*reshape(kron(beta11',ones([90,7])),90,7,2), 3);
sigma12 = std(residual,1,'all');
c12 = sigma12*log(7)/sqrt(7);
[Ghat12,grphat12,~]  = tpwd_pureGFE(residual,c12,2);
%Obtain final estimates
[beta12,alphahat12] =  FE_reg_withcov(Y,X,grphat12,true);

%Make Figure 1
plot(c_grid, model_selec1,'-',color='blue');
xline(c1, '-', color='blue');
hold on
plot(c_grid, model_selec2,'--',color='green');
xline(c2, '--', color='green');
hold on
plot(c_grid, model_selec3,':',color='red');
xline(c3, ':', color='red');
hold on
plot(c_grid, model_selec4,'-.',color='magenta');
xline(c4, '-.', color='magenta');
hold off
xlabel('Thresholding parameter','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
fig = gcf;
fig.Units               = 'centimeters';
fig.Position(3)         = 12;
fig.Position(4)         = 9;
hgexport(fig, 'application_income_dem_pwd_combo.eps');

% cumulative effects
cum0 = beta0(2)/(1-beta0(1))
cum1 = beta1(2)/(1-beta1(1))
cum2 = beta2(2)/(1-beta2(1))
cum3 = beta3(2)/(1-beta3(1))
cum4 = beta4(2)/(1-beta4(1))
cum8 = beta8(2)/(1-beta8(1))
cum9 = beta9(2)/(1-beta9(1))

% Standard Errors
[A,~] = compute_GFE_analytical_cov(grphat1,Y,X,beta1,alphahat1);
avbeta1 = [sqrt(A(1,1)) sqrt(A(2,2))]
grad = [beta1(2)/(1-beta1(1))^2 1/(1-beta1(1))];
avcum1 = sqrt(grad*A*grad')

[A,~] = compute_GFE_analytical_cov(grphat2,Y,X,beta2,alphahat2);
avbeta2 = [sqrt(A(1,1)) sqrt(A(2,2))]
grad = [beta2(2)/(1-beta2(1))^2 1/(1-beta2(1))];
avcum2 = sqrt(grad*A*grad')

[A,~] = compute_GFE_analytical_cov(grphat3,Y,X,beta3,alphahat3);
avbeta3 = [sqrt(A(1,1)) sqrt(A(2,2))]
grad = [beta3(2)/(1-beta3(1))^2 1/(1-beta3(1))];
avcum3 = sqrt(grad*A*grad')

[A,~] = compute_GFE_analytical_cov(grphat4,Y,X,beta4,alphahat4);
avbeta4 = [sqrt(A(1,1)), sqrt(A(2,2))]
grad = [beta4(2)/(1-beta4(1))^2 1/(1-beta4(1))];
avcum4 = sqrt(grad*A*grad')

[A,~] = compute_GFE_analytical_cov(grphat8,Y,X,beta8,alphahat8);
avbeta8 = [sqrt(A(1,1)), sqrt(A(2,2))]
grad = [beta8(2)/(1-beta8(1))^2 1/(1-beta8(1))];
avcum8 = sqrt(grad*A*grad')

[A,~] = compute_GFE_analytical_cov(grphat9,Y,X,beta9,alphahat9);
avbeta9 = [sqrt(A(1,1)), sqrt(A(2,2))]
grad = [beta9(2)/(1-beta9(1))^2 1/(1-beta9(1))];
avcum9 = sqrt(grad*A*grad')




