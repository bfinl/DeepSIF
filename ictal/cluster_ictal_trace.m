% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cluster Ictal Signal
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REFERENCE Kameneva2017_Article_NeuralMassModelsAsAToolToInves.pdf
%% First Identify the B,G value range
clear
A = '5.50';
p_m = '90';
p_v = '120';
all_files = dir(['p_m_' p_m '_p_v_' p_v '_a_' A '*.mat']);
b_all = 2:2:58; % parameter range for B
g_all = 2:2:28; % parameter range for G
label = zeros(length(g_all),length(b_all));
load(dir(['p_m_' p_m '_p_v_' p_v '_a_' A '*b_58_c_135_g_26*.mat']).name, 'data'); 
type1_template = squeeze(data(2001:end,2,:)-data(2001:end,3,:)-data(2001:end,4,:));
for i=1:length(all_files)
    load(all_files(i).name);
    g_id = find(g_all == G);                                                % or mod(i-1, 14)+ 1;  14 G values
    b_id = find(b_all == B);                                           
    d = squeeze(data(2001:end,2,:)-data(2001:end,3,:)-data(2001:end,4,:));
    no_mean_d = d-mean(d);
    
    [ppower,freq] = periodogram(no_mean_d',[],[],2000);
    line_length = mean(abs(no_mean_d(3:2:end)-no_mean_d(1:2:end-2)));   
    
%     % find the frequency range
%     freq10 = minind(abs(freq-10));
%     freq20 = minind(abs(freq-20));
%     freq30 = minind(abs(freq-30));

    if line_length < 0.08                                                  
        if corr(type1_template(1:8000), d(1:8000)) > 0.6  % offset 
            label(g_id, b_id) = 1;                                          % LABEL 1
        else
            max_freq = maxind(ppower);
            if  std(ppower(max(1,max_freq-50):max_freq+50)) > 0.04           
                label(g_id, b_id) = 5;                                      % LABEL 5 Low−voltage rapid activity
            else
                label(g_id, b_id) = 4;                                      % LABEL 4 Slow rhythmic activity
            end
        end
    else
        if freq(maxind(ppower)) > 7   % the max power freq                  
            label(g_id, b_id) = 6;                                          % LABEL 6 Slow quasi−sinusoidal activity
        else
            maxv = max(d);
            if sum(d > 0.9*maxv & islocalmax(d)) > 25                       % count the spikes: THIS 25 DEPENDS ON TIME LENGTH (current length 20000)
                label(g_id, b_id) = 3;                                      % LABEL 3 Sustained discharge of spikes
            else
                label(g_id, b_id) = 2;                                      % LABEL 2 Sporadic spikes
            end
        end
    end
            
end
%% Plot the label
% Review the signals and labels for signals on near the transition
% points
figure
imagesc(label(end:-1:1,:))
set(gca, 'YTick', 1:14, 'YTickLabel', g_all(end:-1:1))  
set(gca, 'XTick', 1:29, 'XTickLabel', b_all)  
colorbar('YTick', 1:6)
%%
save('label_p_m_90_p_v_120_a_5.50_c_135_aa_0.10_bb_0.05.mat','A','g_all','label','p_m','p_v')
%% Setup new Python Simulations
% Based on the classcifcation for B,G, creat the parameter grid for B,G,a,b
% parameter for new ictal nmm simulations
load('label_p_m_90_p_v_120_a_5.50_c_135_aa_0.10_bb_0.05.mat')
lbs = [];
for id = 3:6
[r, c] = ind2sub(size(label), find(label == id));
lbs = cat(1, lbs, [g_all(r)' b_all(c)' id*ones(size(r))]);
end

all_params = [];
[X,Y] = meshgrid([0.05:0.01:0.14], [0.025:0.005:0.070]);
for i = 1:length(lbs)
all_params = cat(1, all_params, [repmat(lbs(i,:), 100, 1) X(:) Y(:)]);
end
% save('all_params_BGab.mat','all_params')
%% Plot newly generated data, identify the parameters we want for a,b
load('all_params_BGab.mat')
A = '5.50';
p_m = '90';
p_v = '120';
all_files = dir(['p_m_' p_m '_p_v_' p_v '_a_' A '*.mat']);
set(0,'DefaultAxesFontSize',10)
xsubplot = 10;
ysubplot = 10;
for i=1:length(all_params)
    load(sprintf('p_m_90_p_v_120_a_5.50_b_%02.0f_c_135_g_%02.0f_aa_%0.3f_bb_%0.3f.mat', all_params(i,2),all_params(i,1),all_params(i,4),all_params(i,5)));
    d = squeeze(data(2001:8000,2,:)-data(2001:8000,3,:)-data(2001:8000,4,:));
    if mod(i-1, xsubplot*ysubplot) == 0
        figure('visible','off')
        sgtitle(sprintf('b-%.3f-g-%.3f', B, G))
        set(gcf,'Position',[1921,41,1920,963])
    end
    subplot(xsubplot,ysubplot,mod(i-1, xsubplot*ysubplot)+1)
% plot data
    % plot(d)
    % if all_params(i,3) = 1
    %     title('background')
    % else
    %     title(sprintf('a-%.3f-b-%.3f', a, b))
    % end

% ----
% plot freq
    [ppower,freq] = periodogram(d'-mean(d),[],[],2000);
    plot(freq(1:300),ppower(1:300))
    if max(ppower) < 0.5
       all_params(i,3) = 1; 
    end
    title(sprintf('a-%.3f-b-%.3f', a, b))
% -----

    if mod(i, xsubplot*ysubplot) == 0
        saveas(gcf, sprintf('figure/freq_b-%.3f-g-%.3f.jpg', B, G))
        close all
    end
end
save('all_params_BGab.mat','all_params')
%% 
% Similar to the previous process, add different g value to the simulation,
% then examine the g parameter range, the parameter ranges can be referred
% to file 'all_params_all_with_g.mat'
%% With g value
A = '5.50';
p_m = '90';
p_v = '120';
all_files = dir(['p_m_' p_m '_p_v_' p_v '_a_' A '*.mat']);
set(0,'DefaultAxesFontSize',10)
xsubplot = 10;
ysubplot = 8;
% load('F:\ProjectsData\JR_SIM\ictalSim\ALL_PARAMS\figure\all_params_all_with_g.mat')
load('figure\all_params_updateg.mat')
for i=201:length(all_params)
    if all_params(i,3) > 1
        load(sprintf('p_m_90_p_v_120_a_5.50_b_%02.0f_c_135_g_%02.0f_aa_%0.3f_bb_%0.3f_gg_%0.3f.mat', all_params(i,2),all_params(i,1),all_params(i,4),all_params(i,5),all_params(i,6)));
        d = squeeze(data(2001:8000,2,:)-data(2001:8000,3,:)-data(2001:8000,4,:));
        if mod(i-1, xsubplot*ysubplot) == 0
            figure('visible','off')
            sgtitle(sprintf('b-%.3f-g-%.3f', B, G))
            set(gcf,'Position',[1921,41,1920,963])
        end
        subplot(xsubplot,ysubplot,mod(i-1, xsubplot*ysubplot)+1)
        % plot data
            % plot(d)
            % if all_params(i,3) == 1
            %     title('background')
            % else
            %     title(sprintf('a-%.3f-b-%.3f', a, b))
            % end

        % ----
        % plot freq
            [ppower,freq] = periodogram(d'-mean(d),[],[],2000);
            plot(freq(1:300),ppower(1:300))
            [max_p, max_ind] = max(ppower);
            if max_p< 0.5 && freq(max_ind) < 15
            all_params(i,3) = 1; 
            end
            title(sprintf('b-%.3f-g-%.3f-a-%.3f-b-%.3f', B, G, a, b))

        if mod(i, xsubplot*ysubplot) == 0
            sprintf('figure/freq_b-%.3f-g-%.3f.jpg', B, G)
            saveas(gcf, sprintf('figure/freq_b-%.3f-g-%.3f.jpg', B, G))
            close all
%             save('all_params_final.mat','all_params')
        end

    end
end


%% EXTRACT and SAVE DATA
all_data = [];
for i=1:length(all_params)
    if all_params(:,3) ~= 1
        load(sprintf('p_m_90_p_v_120_a_5.50_b_%02.0f_c_135_g_%02.0f_aa_%0.3f_bb_%0.3f.mat', all_params(i,2),all_params(i,1),all_params(i,4),all_params(i,5)));
        d = squeeze(data(2001:end,2,:)-data(2001:end,3,:)-data(2001:end,4,:));
        all_data = [all_data d];
    end
end
save('all_data.mat','all_data')

%%
