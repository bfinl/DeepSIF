% Evaluate estimated sources based on saved resource after performing otsu
% threshold.
clear
dataset_name = '_sample_source2';
fname = '_sample_source2';
model_id = '148';
gt = load(['../source/Simulation/test' dataset_name '.mat']);             % ground truth
num_source = size(gt.selected_region, 2);
load(['../model_result/' model_id '_the_model/model_best.pth.tar_preds_test' fname '.mat']);

load('../anatomy/dis_matrix_fs_20k.mat')
all_dis = raw_dis_matrix;
% Variabled loaded:
% FROM source
% gt.selected_region   : array; ground truth source region, start from 0, num_examples * num_source * MAX_SIZE 
%                    (MAX_SIZE=70, the number of cortical regions in each 
%                    example is different, padd with 15213 to size 70
% FROM model_result
% all_regions          : cell; DeepSIF reconstructed source region, start from 0, num_examples * 1
% all_out              : cell; activity in DeepSIF reconstructed source region; num_examples * 1
% all_num              : array: ground truth activity; num_examples * num_source * num_time
% FROM anatomy
% all_dis              : distance between cortical source regions
% all_area             : the area of each cortical source region

%%
speci = nan(length(all_out), num_source);
sensiti = nan(length(all_out), num_source);
le = nan(length(all_out), num_source);
all_corr = nan(length(all_out), num_source);

recon_regions = cell(length(all_out), num_source);
recon_activity = cell(length(all_out), num_source);

for i= 1:length(all_out)
    
    % gather all source regions, remove padded variable.
    all_label = reshape(squeeze(gt.selected_region(i,:,:))',[],num_source);
    num_region_per_source = sum(~myisnan(all_label),1);
    all_label(myisnan(all_label)) = [];
    % recon regions
    current_regions = all_regions{i};
    if isempty(current_regions)
        continue
    end
    
    % assign the each recon source region to its closest source patch
    [~, min_ind] = min(all_dis(all_label+1,current_regions+1),[],1);
    mapping = []; % size: 1 * total_ground_truth_source_regions
    for ii=1:length(num_region_per_source)
        mapping = [mapping ii*ones(1,num_region_per_source(ii))];
    end
    source_id = mapping(min_ind);
    
    % calculate metrics
    for k=1:max(source_id)
        recon = current_regions(source_id==k);
        lb =  squeeze(gt.selected_region(i,k,:));
        lb(myisnan(lb)) = [];
        if ~isempty(recon)
            recon_regions{i,k} = recon; 
            recon_activity{i,k} = all_out{i}(source_id==k,:);
            
            [se, sp] = get_sesp(recon+1, lb'+1, all_dis, all_area);
            
            interc = intersect(recon,lb);
            speci(i,k) = sp;
            sensiti(i,k) = se;
            all_corr(i,k) = corr(mean(all_out{i}(source_id==k,:),1)',squeeze(all_nmm(i,k,:)));
            le(i,k) = (mean(min(all_dis(recon+1,lb+1),[],2)) + mean(min(all_dis(recon+1,lb+1),[],1)))/2;
        end
    end    
        
end
% save and display results
s(1) = mean(speci(:), 'omitnan');s(2) = mean(sensiti(:),'omitnan');s(3) = mean(all_corr(:),'omitnan');s(4) = mean(le(:),'omitnan');s
save(['../model_result/' model_id '_the_model/recon' fname '.mat'],'speci','sensiti','le','all_corr','recon_regions','recon_activity')
%%
function y = myisnan(x)
    y = abs(x-15213)<1e-6;
end

function [se, sp] = get_sesp(selected_recon, gt_region, all_dis, all_area)
% Get the sensitivity and specificity based on the definition from ( Chowdhury, et al., Plos One, 2013)    
% Input:
% selected_recon       : estimated source region id, start with 1
% gt_region            : simulated source region id, start with 1
% all_dis              : distance between cortical source regions
% all_area             : the area of each cortical source region

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
    
% away from the ground truth: AUC far
      outside_source = setdiff(selected_recon, [near_source gt_region]);
      far_source = find(min(all_dis(outside_source,:),[],1) < 30);
      
     
      if length(far_source) >= length(gt_region)
        not_gt_2 = far_source(randperm(length(far_source)));
        not_gt_2 = not_gt_2(1:length(gt_region));                                    
      else
        others = setdiff(1:994,[gt_region near_source far_source]);
        others = others(randperm(length(others)));
        not_gt_2 = [far_source others(1:(length(gt_region)-length(far_source)))];
      end


    notselected_recon = setdiff(1:994, selected_recon);
    se = sum(all_area(intersect(gt_region, selected_recon)))/sum(all_area(gt_region));
%     se = length(intersect(gt_region, selected_recon))/length(gt_region);

    overlap_not1 = intersect(not_gt_1, notselected_recon);
    overlap_not2 = intersect(not_gt_2, notselected_recon);
    sp1 = sum(all_area(overlap_not1))/sum(all_area(not_gt_1));
    sp2 = sum(all_area(overlap_not2))/sum(all_area(not_gt_2));
    sp = (sp1+sp2)/2;
end

