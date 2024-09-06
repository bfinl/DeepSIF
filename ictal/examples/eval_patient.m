brainstorm
%%
clear
patient_id = 'P2';
%% %%%%%%%%%%%%%%% load anatomy information %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([patient_id '/anatomy/cortex_fs_in_curryspace.mat']);
load([patient_id '/anatomy/projected_resection_to_fs_cortex.mat']);
load([patient_id '/rnn_test.mat'])
load(['../../anatomy/dis_matrix_fs_20k.mat'])

% eloc = load(['../../anatomy/electrode_75.mat'])
% eloc = eloc.eloc;

%% %%%%%%%%%%%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
all_data = [];
for i=1:3
    load([patient_id '/sz_data/data' int2str(data_range(i)) '.mat'])
    all_data = [all_data;data];
end
data = all_data;
data = bst_bandpass_hfilter(double(data'), 500.0, LOW_FREQ, HIGH_FREQ)'; 
data = data/(max(abs(data(:))));

% plot_data = reshape(permute(all_out(data_range,:,:),[3,2,1]),994,[])';
% plot_data = bst_bandpass_hfilter(double(plot_data'), 500.0, LOW_FREQ, HIGH_FREQ)'; 
% plot_data = permute(reshape(plot_data, 500, [],994),[2,1,3]);
% plot_data = plot_data./max(abs(plot_data),[], [2,3]);
% plot_data = reshape(permute(plot_data,[3,2,1]),994,[])';

%% %%%%%%%%%%%%%%% load results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate signal norm
% if exist('J_norm','var') == 0
%     n=5;
%     J_norm = zeros(n, size(pos,1));
%     J_square = zeros(n, size(pos,1));
%     load(['rnn_test_' SZ_ID '_75_10171_ar.mat'], 'LOW_FREQ', 'HIGH_FREQ')  
%     filter_recon = reshape(permute(all_out,[3,2,1]),994,[])'; % all_out: num of 1s data, time, num_source_region
%     recon2 = bst_bandpass_hfilter(double(filter_recon'), 500.0, LOW_FREQ, HIGH_FREQ)';  
%     % Reshape the results
%     WINDOW_SIZE = 500;
%     all_out2 = permute(reshape(recon2, WINDOW_SIZE, [],994),[2,1,3]);

%     norm_recon = squeeze(mynorm(all_out2(1:n,:,:), 2));
%     parfor i=1:n
%         for k =1:994
%             J_norm(i,:) = J_norm(i,:) + (rm == (k-1))*abs(norm_recon(i,k));%*abs(data(max_time(i),preds{i}(k)+1));
%             J_square(i,:) = J_square(i,:) + (rm == (k-1))*abs(norm_recon(i,k).^2);
%         end
%     end
%     save(['rnn_test_' SZ_ID '_75_' int2str(model_id) '_ar.mat'],'LOW_FREQ', 'HIGH_FREQ','J_norm','J_square', '-append')
% end

J = J_norm;
J = J./max(J,[],2);

%% %%%%%%%%%%%%%%% Calculate LE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
projected_resection = pos(find(ismember(rm,resection_region)),:);
gt_area = sum(all_area(resection_region+1));

Jrecon= nanmean(J([1:3],:),1);
region_J = zeros(1,994);
for kk =1:994
    region_J(kk) = mean(Jrecon(ismember(rm, kk-1)));   % tp_rg.
end
region_J = region_J-min(region_J);
region_J = region_J/max(region_J);

Jrecon=Jrecon/max(abs(Jrecon));
thre=graythresh(Jrecon);
ind=find(Jrecon>thre);
pred_region = unique(rm(ind));
pred_area = sum(all_area(pred_region+1));
%         save(['rnn_test_' SZ_ID '_75_' int2str(model_id) '_ar.mat'], 'pred_region', '-append')


overlap_area = sum(all_area(intersect(resection_region+1,pred_region+1)));
[auc, se, sp, thres] = get_auc(region_J, resection_region+1, raw_dis_matrix, all_area); 
sensitivity = se(minind(abs(thres-thre)));
specificity = sp(minind(abs(thres-thre)));

w = Jrecon(ind)/sum(Jrecon(ind));
sz_le(1)=sum(min(mydis(projected_resection',pos(ismember(rm,rm(ind)),:)'),[],1).*w);
sz_le(2) = sum(min(mydis(projected_soz',pos(ismember(rm,rm(ind)),:)'),[],1).*w); % Method 3
sz_le(3) = mean(min(mydis(projected_soz',pos(ismember(rm,rm(ind)),:)'),[],2)); % SOZ to all recon

[sensitivity, specificity sz_le]
%% %%%%%%%%%%%%%%% plot result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
visualize_result(pos,tri,Jrecon,'FaceAlpha',0.5,'thre',0);  % function defined in 'misc_scripts'



function [Q, se, sp, thres] = get_auc(J_recon, gt_region, all_dis, all_area)
    % gt_region start with 1
    % Chowdhury Plos One 2013    
    % close to the ground truth: AUC_close
        near_source = setdiff(find(min(all_dis(gt_region,:),[],1) < 30), gt_region);
        if length(near_source) >= length(gt_region)
            not_gt_1 = near_source(randperm(length(near_source)));
            not_gt_1 = not_gt_1(1:length(gt_region));                                    
        else
            others = setdiff(1:994,[gt_region near_source]);
            others = others(randperm(length(others)));
            not_gt_1 = [near_source others(1:(length(gt_region)-length(near_source)))];
        end
    % not close to the ground truth: AUC far
          tmp_J_recon = J_recon;
          tmp_J_recon([gt_region near_source]) = 0;
          far_source = find(min(all_dis(maxind(tmp_J_recon),:),[],1) < 30);
          if length(far_source) >= length(gt_region)
            not_gt_2 = far_source(randperm(length(far_source)));
            not_gt_2 = not_gt_2(1:length(gt_region));                                    
          else
            others = setdiff(1:994,[gt_region near_source far_source]);
            others = others(randperm(length(others)));
            not_gt_2 = [far_source others(1:(length(gt_region)-length(far_source)))];
          end
    %     not_gt_2 = setdiff(1:994,[gt_region not_gt_1]);
    %     not_gt_2 = not_gt_2(randperm(length(not_gt_2)));
    %     not_gt_2 = not_gt_2(1:length(gt_region));  
    
        thres = [0:0.001:1];%logspace(0,log10(2),20)-1;%0:0.05:1;
        se = zeros(1,length(thres));
        sp = zeros(1,length(thres));
        for i=1:length(thres)
            selected_recon = find(J_recon>=thres(i));
            notselected_recon = find(J_recon<thres(i));
            se(i) = sum(all_area(intersect(gt_region, selected_recon)))/sum(all_area(gt_region));
            
            overlap_not1 = intersect(not_gt_1, notselected_recon);
            overlap_not2 = intersect(not_gt_2, notselected_recon);
            sp1 = sum(all_area(overlap_not1))/sum(all_area(not_gt_1));
            sp2 = sum(all_area(overlap_not2))/sum(all_area(not_gt_2));
            
            sp(i)= (sp1+sp2)/2;% (sp1+sp2)/2;  %sp2;%
        end
        Q = trapz(se,sp);
    end