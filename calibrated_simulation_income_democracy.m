% Simulation calibrated to the Income/Democracy application
rng('default');
B = 1000;
N = 74;
T = 7;
%{
% Discrete UH (Figures)
G_seq = [3 5 11];
sigma_sq_seq = [0.0940 0.0821 0.0634];
c_seq = linspace(0.01,2,100); 

res = zeros(length(G_seq),length(c_seq),6,B);

for g=1:3
    noise = normrnd(0, sqrt(sigma_sq_seq(g)),B,N,T);
    true_grp = xlsread(strcat(string(G_seq(g)),'_calib_exercise.xlsx'),2);
    dist_true = abs(bsxfun(@minus,true_grp,true_grp'))>0;
    true_grp_effects = reshape(xlsread(strcat(string(G_seq(g)),'_calib_exercise.xlsx'),3), 1, G_seq(g));
    true_signal = zeros(N,1);
    for n=1:N
        true_signal(n,1) = true_grp_effects(true_grp(n));
    end
    for b=1:B
        Y = reshape(repelem(true_signal, T),T,N)' + reshape(noise(b,1:N,1:T), N, T);
        c_id = 1;
        for c=c_seq
            [G_hat,grp,grp_effects] = pwd_estimators.pwd1(Y,c);
            HD = HausdorffDist(grp_effects, true_grp_effects');
            dist_hat = abs(bsxfun(@minus,grp,grp'))>0;
            FP = sum((1-dist_hat).*dist_true, 'all')/2;
            TP = (sum((1-dist_hat).*(1-dist_true), 'all')-n)/2;
            FN = sum(dist_hat.*(1-dist_true), 'all')/2;
            TN = sum(dist_hat.*dist_true, 'all')/2;
            P = TP/(TP+FP);
            R = TP/(FN+TP);
            RI =(TP+TN)/(TP+FP+FN+TN); 
            res(g,c_id,:, b) = [G_hat,(G_hat==G_seq(g)),HD,RI,P,R];
            c_id = c_id + 1;
        end
    end
end
export_res_fig = mean(res, 4);
save('figure_thresh_sensitiv_res_calib_mc.mat','export_res_fig');


% plot results
thresh_plot(export_res_fig,1,'$\hat G$',c_seq,0,15)
thresh_plot(export_res_fig,2,'Probability of the event $\hat G=G$',c_seq,0,1)
thresh_plot(export_res_fig,3,'HD',c_seq,0,2)
thresh_plot(export_res_fig,4,'RI',c_seq,0,1)
thresh_plot(export_res_fig,5,'P',c_seq,0,1)
thresh_plot(export_res_fig,6,'R',c_seq,0,1)
%}


% Continuous UH (do we observe some phase transition?)
ymean = xlsread(strcat(string(3),'_calib_exercise.xlsx'),5);
yvar  = xlsread(strcat(string(3),'_calib_exercise.xlsx'),6);
%{
% Setting 1
true_signal = normrnd(ymean, sqrt(yvar/2),B,N);
%}
% Setting 2
ber = (rand(B,N)<0.5);
z = normrnd(0, sqrt(0.05),B,N);
true_signal = ber.*abs(1-z) + (1-ber).*abs(z);
noise = normrnd(0, 0.001,B,N,T);
res = zeros(length(c_seq),3,B);
c_seq = linspace(0.01,2,100); 

for b=1:B
    Y = reshape(repelem(true_signal(b,:),T),T,N)' + reshape(noise(b,1:N,1:T), N, T);
    Y = Y.*(Y<=ones(N,T)) + (Y>ones(N,T));
    c_id = 1;
    for c=c_seq
        [G_hat,grp,grp_effects] = pwd_estimators.pwd1(Y,c);
        HD = HausdorffDist(grp_effects, true_signal(b,:)');
        res(c_id,:, b) = [G_hat,(G_hat==G_seq(g)),HD];
        c_id = c_id + 1;
    end
end

export_res_fig = mean(res, 4);
t = tiledlayout(1,2);
nexttile
plot(c_seq,reshape(export_res_fig(:,1),1, length(c_seq)));
ylim([0 N])
xlim([0 2])
xlabel('$c$','Interpreter','latex') 
ylabel('$\hat G$','Interpreter','latex') 
nexttile
plot(c_seq,reshape(export_res_fig(:,3),1, length(c_seq)));
%ylim([0 2])
xlim([0 2])
xlabel('$c$','Interpreter','latex') 
ylabel('HD','Interpreter','latex') 
%exportgraphics(t,'figure_thresh_sensitiv_res_calib_mc_cont_UH1.eps')
exportgraphics(t,'figure_thresh_sensitiv_res_calib_mc_cont_UH2.eps')




% Create plots for each feature
function thresh_plot(data,feature,ylab,c_seq,yl,yu)
    t = tiledlayout(1,3);
    nexttile
    plot(c_seq,reshape(data(1,:,feature),1, length(c_seq)));
    ylim([yl yu])
    xlim([0 2])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=3$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(2,:,feature),1, length(c_seq)));
    ylim([yl yu])
    xlim([0 2])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=5$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(3,:,feature),1, length(c_seq)));
    ylim([yl yu])
    xlim([0 2])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=11$','Interpreter','latex')
    exportgraphics(t,'figure_thresh_sensitiv_res_calib_mc'+string(feature)+'.eps')
end
