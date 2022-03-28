% APPLICATION ACEMOGLU & AL (2008)
% load and clean data (http://economics.mit.edu/files/5000)
M = csvread('outcome.csv');
M = rmmissing(M); 
[N, T] = size(M);

% Plot data
histogram(M,20);

% Compute estimates
prec = 100;
c_seq = linspace(0.01,2,prec);
Gres = zeros(prec,1);
Gres_t = zeros(prec,1);
grp_effects_res = {};
grp_effects_res_t = {};
grp_memberships = {};
grp_memberships_t = {};
i = 1;
for c=c_seq
    [G_hat,grp_labels,grp_effects] = pwd_estimators.pwd(M,c);
    [G_hat_t,grp_labels_t,grp_effects_t] = pwd_estimators.tpwd(M,c,false,false);
    Gres(i,1) = G_hat;
    Gres_t(i,1) = G_hat_t;
    grp_effects_res{end+1} = grp_effects;
    grp_effects_res_t{end+1} = grp_effects_t;
    grp_memberships{end+1} = grp_labels;
    grp_memberships_t{end+1} = grp_labels_t;
    i = i + 1;
    disp(i)
end

% Output Figure 1
t = tiledlayout(2,1);
nexttile
plot(c_seq, Gres(:,1));
xline(log(7)/sqrt(7),'--','$\log{7}/\sqrt{7}$','Interpreter','latex');
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex') ;
nexttile
scatter(repelem(c_seq(1),length(grp_effects_res{1})),grp_effects_res{1}); 
hold on;
for c=2:length(c_seq)
    scatter(repelem(c_seq(c),length(grp_effects_res{c})),grp_effects_res{c}); 
    hold on;
end
xline(log(7)/sqrt(7),'--','$\log{7}/\sqrt{7}$','Interpreter','latex');
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated group effects ','Interpreter','latex') ;
exportgraphics(t,'application_income_dem_pwd.eps')

% Output Figure 2
t = tiledlayout(2,1);
nexttile
plot(c_seq, Gres_t(:,1));
xline(log(7)/sqrt(7),'--','$\log{7}/\sqrt{7}$','Interpreter','latex');
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated number of groups','Interpreter','latex');
nexttile
exog = repelem(grp_memberships_t{1},T);
scatter3(repelem(1:T,N)',repelem(c_seq(1), N*T)',reshape(M',N*T,1),ones(N*T, 1),exog); 
hold on;
for c=2:length(c_seq)
    exog = repelem(grp_memberships_t{c},T);
    scatter3(repelem(1:T,N)',repelem(c_seq(c), N*T)',reshape(M',N*T,1), ones(N*T, 1),exog); 
    hold on;
end
%xline(log(7)/sqrt(7),'--','$\log{7}/\sqrt{7}$','Interpreter','latex');
xlabel('Threshold','Interpreter','latex');
ylabel('Estimated group effects ','Interpreter','latex') ;
exportgraphics(t,'application_income_dem_tpwd.eps')

% Output parameters for the calibrated application
for j=[44 40 34]
    grp_labels = grp_memberships{j};
    % build group prediction
    exog = dummyvar(grp_labels); 
    exog = repmat(exog',T,1);
    exog = reshape(exog,[],N*T)';
    endog_hat = exog * grp_effects_res{j};
    endog = reshape(M',N*T,1);
    xlswrite(strcat(string(Gres(j,1)),'_calib_exercise.xlsx'),grp_memberships{j},['sheet_' num2str(1)]);
    xlswrite(strcat(string(Gres(j,1)),'_calib_exercise.xlsx'),grp_effects_res{j},['sheet_' num2str(2)]);
    xlswrite(strcat(string(Gres(j,1)),'_calib_exercise.xlsx'),mean((endog-endog_hat).^2),['sheet_' num2str(3)]);
    xlswrite(strcat(string(Gres(j,1)),'_calib_exercise.xlsx'),mean(endog),['sheet_' num2str(4)]);
    xlswrite(strcat(string(Gres(j,1)),'_calib_exercise.xlsx'),var(endog),['sheet_' num2str(5)]);
    xlswrite(strcat(string(Gres(j,1)),'_calib_exercise.xlsx'),c_seq(j),['sheet_' num2str(6)]);   
end
