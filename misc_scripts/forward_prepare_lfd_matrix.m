% Calculate the forward matrix based on the cortical segmentation.
% Assume the headmodel was exported from Brainstorm as bs_headmodel

assert(size(bs_headmodel.Gain, 2) ==  length(rm)*3)

lfd_free = bs_headmodel.Gain;
lfd = lfd_free2fix(lfd_free,bs_headmodel.GridOrient);
fwd = fwd_to_rmfwd(lfd, rm);

save('../anatomy/leadfield_test.mat','fwd')

function fwd = lfd_free2fix(lfd, ori)
% Transfer the free direction leadfield to fix direction
% INPUTS:
%     - lfd        : free direction leadfield, num_electrode * (num_vertice*3)
%     - ori        : orientation, num_vertice * 3
% OUTPUTS:
%     - fwd        : free direction leadfield, num_electrode * num_vertice

num_vertices = size(lfd,2)/3;
fwd = zeros(size(lfd,1), num_vertices);
for i = 1:num_vertices
    fwd(:,i) = lfd(:, (i-1)*3+1:i*3) * ori(i,:)';
end
end

function new_fwd = fwd_to_rmfwd(fwd, rm)
% Calculate the sumed forward matrix for each region, assume each dipole in
% the source region has the same activity
% INPUTS:
%     - fwd     : fix direction leadfield, num_electrode * num_vertice
%     - rm      : region mapping array, map each vertice to a region, num_vertic*1
% OUTPUTS:
%     - new_fwd : leadfield for each region, num_electrode * num_region

unique_rm = unique(rm);
new_fwd = zeros(size(fwd,1), length(unique_rm));
for i=1:length(unique_rm)
    new_fwd(:,i) = sum(fwd(:, rm==unique_rm(i)),2);
end
end