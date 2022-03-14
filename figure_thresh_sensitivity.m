% Reproducibility
rng('default');

% Monte Carlo parameters
B = 1000;
n = 120;
T_seq = ceil(linspace(sqrt(n), n, 3));
c_seq = linspace(0.1,20,40); 
G_seq = [2 3 4];
sig_seq = [1/4 1/2 1];
res = zeros(3, length(T_seq),length(c_seq),5,B,3);

% Run simulations
sig_id = 1;
for sigma=sig_seq
    noise = normrnd(0, sigma, B, n, n);
    g = 1;
    for G=G_seq
    true_grp = repelem([1:G],n/G)';
    dist_true = abs(bsxfun(@minus,true_grp,true_grp'))>0;
    true_grp_effects = linspace(-G/2,G/2,G);
    true_signal = repelem(true_grp_effects,n/G);
    t = 1;
    for T=T_seq
        for b=1:B
            Y = reshape(repelem(true_signal, T),T,n)' + reshape(noise(b,1:n,1:T), n, T);
            c_id = 1;
            for c=c_seq
                [G_hat,grp,grp_effects] = pwd_estimators.pwd1(Y,c*log(T)/sqrt(T));
                HD = HausdorffDist(grp_effects, true_grp_effects');
                dist_hat = abs(bsxfun(@minus,grp,grp'))>0;
                FP = sum((1-dist_hat).*dist_true, 'all')/2;
                TP = (sum((1-dist_hat).*(1-dist_true), 'all')-n)/2;
                FN = sum(dist_hat.*(1-dist_true), 'all')/2;
                TN = sum(dist_hat.*dist_true, 'all')/2;
                P = TP/(TP+FP);
                R = TP/(FN+TP);
                RI =(TP+TN)/(TP+FP+FN+TN); 
                res(g, t, c_id, :, b, sig_id) = [G_hat, HD, RI, P, R];
                c_id = c_id + 1;
            end
        end
        t = t + 1
    end
    g = g + 1
    end
    sig_id = sig_id + 1
end
export_res_fig = mean(res, 5);
%save('figure_thresh_sensitiv_res.mat','export_res_fig');

% plot results
thresh_plot(export_res_fig,1,'$\hat G$',c_seq,0,10)
thresh_plot(export_res_fig,2,'HD',c_seq,0,2)
thresh_plot(export_res_fig,3,'RI',c_seq,0,1)
thresh_plot(export_res_fig,4,'P',c_seq,0,1)
thresh_plot(export_res_fig,5,'R',c_seq,0,1)


% Create plots for each feature
function thresh_plot(data,feature,ylab,c_seq,yl,yu)
    t = tiledlayout(3,3);
    nexttile
    plot(c_seq,reshape(data(1,1,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(1,1,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(1,1,:,feature,3),1, length(c_seq)),':');
    %yline(2,'-','$G=2$','Interpreter','latex');
    legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=2,N=120,T=11$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(1,2,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(1,2,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(1,2,:,feature,3),1, length(c_seq)),':');
    %yline(2,'-','$G=2$','Interpreter','latex');
    ylim([yl yu])
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=2, N=120, T=66$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(1,3,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(1,3,:,feature,2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(1,3,:,feature,3),1, length(c_seq)),':');
    %yline(2,'-','$G=2$','Interpreter','latex');
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=2,N=120, T=120$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(2,1,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(2,1,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(2,1,:,feature,3),1, length(c_seq)),':');
    %yline(3,'-','$G=3$','Interpreter','latex');
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=3, N=120, T=11$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(2,2,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(2,2,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(2,2,:,feature,3),1, length(c_seq)),':');
    %yline(3,'-','$G=3$','Interpreter','latex');
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=3,N=120, T=66$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(2,3,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(2,3,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(2,3,:,feature,3),1, length(c_seq)),':');
    %yline(3,'-','$G=3$','Interpreter','latex');
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=3,N=120, T=120$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(3,1,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(3,1,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(3,1,:,feature,3),1, length(c_seq)),':');
    %yline(3,'-','$G=4$','Interpreter','latex');
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab, 'Interpreter','latex') 
    title('$G=4,N=120, T=11$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(3,2,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(3,2,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(3,2,:,feature,3),1, length(c_seq)),':');
    %yline(3,'-','$G=4$','Interpreter','latex');
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=4,N=120, T=66$','Interpreter','latex')
    nexttile
    plot(c_seq,reshape(data(3,3,:,feature,1),1, length(c_seq)),c_seq,...
        reshape(data(3,3,:,feature, 2),1, length(c_seq)),'--', ...
        c_seq, reshape(data(3,3,:,feature,3),1, length(c_seq)),':');
    %yline(3,'-','$G=4$','Interpreter','latex');
    %legend('$\sigma=0.25$','$\sigma=0.5$','$\sigma=1$', 'Interpreter','latex')
    ylim([yl yu])
    xlabel('$c$','Interpreter','latex') 
    ylabel(ylab,'Interpreter','latex') 
    title('$G=4,N=120, T=120$','Interpreter','latex')
    exportgraphics(t,'figure_thresh_sensitiv_res'+string(feature)+'.eps')
end

