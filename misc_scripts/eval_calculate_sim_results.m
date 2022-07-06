% Evaluate estimated sources based on saved resource after performing otsu
% threshold.
clear
dataset_name = '_sample_source2';
fname = '_sample_source2';
model_id = '75';
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

%%
precision = nan(length(all_out), num_source);
recall = nan(length(all_out), num_source);
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
            
            interc = intersect(recon,lb);
            precision(i,k) = length(interc)/length(recon);
            recall(i,k) = length(interc)/length(lb);
            all_corr(i,k) = corr(mean(all_out{i}(source_id==k,:),1)',squeeze(all_nmm(i,k,:)));
            le(i,k) = mean(min(all_dis(recon+1,lb+1),[],2));
        end
    end    
        
end
% save and display results
s(1) = mean(precision(:), 'omitnan');s(2) = mean(recall(:),'omitnan');s(3) = mean(all_corr(:),'omitnan');s(4) = mean(le(:),'omitnan');s
save(['../model_result/' model_id '_the_model/recon' fname '.mat'],'precision','recall','le','all_corr','recon_regions','recon_activity')
%%
function y = myisnan(x)
    y = abs(x-15213)<1e-6;
end



