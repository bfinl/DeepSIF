%% SAVE MATLAB FILE
cd(['F:\DataResearch\' PatientName '\Label'])
mri_name = dir(['F:\DataResearch\Brainstorm\' ProtocolName '\anat\CAT\subjectimage*01.mat']);
sMRI = load([mri_name.folder '\' mri_name.name]);
bs_low = load(['F:\DataResearch\Brainstorm\' ProtocolName '\anat\CAT\tess_cortex_pial_20484V.mat']);
bs_headmodel = load(['F:\DataResearch\Brainstorm\' ProtocolName '\data\CAT\MEG\headmodel_surf_openmeeg.mat']);
SEEG = load(['F:\DataResearch\Brainstorm\' ProtocolName '\data\CAT\Implantation\channel.mat']);
save('bs_exported','sMRI','bs_low','bs_headmodel','SEEG')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD CORTEX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[pos, tri] = fs_get_surf(['F:\DataResearch\' PatientName '\Label\fs'],'fname','h.pial');  % .native
[rm, ~] = fs_get_rm(['F:\DataResearch\' PatientName '\Label\fs'], 'nmm_994');
save(['F:\DataResearch\' PatientName '\Label\fs_cortex.mat'],'pos','tri','rm')
no_resection = isempty(dir(['F:\DataResearch\' PatientName '\Label\resection_curry_image_ras.mat']));
if ~no_resection
    load(['F:\DataResearch\' PatientName '\Label\resection_curry_image_ras.mat'])
end
load(['F:\DataResearch\' PatientName '\Label\bs_exported.mat'],'sMRI')
% bs_resection = cs_convert(sMRI,  'mri', 'scs', curryloc'/1000);
pos = fs_to_bs(pos, sMRI, 'fs_to_bs');
save(['F:\DataResearch\' PatientName '\Label\fs_cortex_bs.mat'],'pos','tri','rm')

%% Prepare Brainstorm LFD
load(['F:\DataResearch\' PatientName '\Label\fs_cortex_bs.mat'],'rm', 'pos')
load(['F:\DataResearch\' PatientName '\Label\bs_exported.mat'],'bs_low')
low_v = bs_low.Vertices;
high_v = bs_high.Vertices;  %pos;
min_ind = zeros(size(low_v,1),1);
min_dis = zeros(size(low_v,1),1);
rm_downsampled = zeros(size(low_v,1),1);
iter = ceil(size(low_v,1)/100);
for i=1:iter
    point_range = ((i-1)*100+1):min(i*100, size(low_v,1));
    [min_dis(point_range), min_ind(point_range)] = min(mydis(high_v', low_v(point_range,:)'),[],1);
    rm_downsampled(point_range) = rm(min_ind(point_range));
end
rm = rm_downsampled;
pos = low_v;
tri = bs_low.Faces;
save(['F:\DataResearch\' PatientName '\Label\fs_cortex_bs_low_res.mat'],'rm','min_ind','pos','tri')
plot_nmm_region(0:993, '998+20k','pos',pos,'tri',tri, 'rm',rm, 'FaceAlpha',0.9)
%% Get fwd
lfd_free = bs_headmodel.Gain(1:306,:);
lfd = lfd_to_fwd(lfd_free,bs_headmodel.GridOrient);
ori = bs_headmodel.GridOrient;
fwd = fwd_to_rmfwd(lfd, rm);
save(['F:\DataResearch\' PatientName '\Label\bs_leadfield.mat'],'lfd','lfd_free','fwd')
save(['F:\DataResearch\' PatientName '\Label\fs_cortex_bs_low_res.mat'],'ori','-append')
fwd_small = fwd(3:3:306,:);
save(['F:\DataResearch\' PatientName '\Label\fwd_' PatientName '.mat'],'fwd_small')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Identify SOZ IDX %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
onset = Parameter.Onset;
spread = Parameter.Spread;
soz_idx = find(ismember({SEEG.Channel.Name}, onset));
spread_idx = find(ismember({SEEG.Channel.Name}, spread));
seeg = [SEEG.Channel.Loc]';
soz = seeg(soz_idx,:);
spread = seeg(spread_idx,:);
% ied = seeg(ismember({SEEG.Channel.Name}, Parameter.Spike),:);
if isempty(dir('projected_resection_to_bs.mat'))
    save('projected_resection_to_bs.mat', 'seeg','soz','spread')
else
    save('projected_resection_to_bs.mat', '-append','seeg','soz','spread')
end

