function visualize_result(pos, tri, varargin)
% input  
%       pos:    num_voxel * 3   
%       tri:    num_tri * 3
%       varargin:   
%                   value:  
%                   source      -- the ground truth, the index of the patch location
%                   FaceAlpha   -- the alpha for the value
%                   save        -- save the figure or not, logical, default, false
%                   row/col     -- use subplot to display

%% ---------- sparse inputs ---------------------------
% Ref: https://www.mathworks.com/help/matlab/matlab_prog/parse-function-inputs.html
p = inputParser;

addRequired(p,'pos',@isnumeric);
addRequired(p,'tri',@isnumeric);
addOptional(p,'value',0, @isnumeric);               % num_example * num_voxel of the reconstruction， if not provided, just plot the cortex shape.
addOptional(p,'source',{[]},@iscell)                % the ground truth, the index of the patch location; if source is provided, it would be plotted as a green overlay covering the cortex
addOptional(p,'FaceAlpha',0.2,@isnumeric)           % the alpha for the reconstruction
addOptional(p,'SourceFaceAlpha',0.2,@isnumeric)     % the alpha for the source
addOptional(p,'save','',@isstr)                     % save the figure or not, logical, default, false
addParameter(p,'row',1,@isnumeric)                  % use subplot to display
addParameter(p,'col',1,@isnumeric)                  % use subplot to display
addParameter(p,'neg',0,@isnumeric)                  % keep the negatives in value
addParameter(p,'green',0,@isnumeric)                % plot value using green color (to plot the ground truth cortex)
addParameter(p,'thre',0.1,@isnumeric)               % if we are thresholding the reconstruction value 
addParameter(p,'view',[-86,17],@isnumeric)          % view angle
addParameter(p,'titles',{0},@iscell)                % figure titles
addParameter(p,'new_fig',1,@isnumeric)
addParameter(p,'normalize',1,@isnumeric)            % normalized reconstruction value to [0,1]
% addParameter(p,'FaceVertexCData',1,@isnumeric)
parse(p, pos, tri, varargin{:})

value = p.Results.value;
% check inputs
if sum(abs(value))>0
    % There are more points than the source (common for outputs from Curry)
    if (size(pos, 1) > size(value, 2)) 
        if size(value, 2) == 1
            value = value';
        else
        % pad the reconstruction
        value = [value zeros(size(value, 1), size(pos, 1)-size(value, 2))];
        end
    end
    if size(value,1)>size(value,2)
    % The first dimension is supposed to be the number of reconstructions, it
    % is usually smaller than the number of dipoles
        value = value';
    end
end
num_example = size(value,1);
img_per_fig = p.Results.row * p.Results.col;
img_ii = 1;

if ~p.Results.new_fig && (p.Results.row ~=1 || p.Results.col ~= 1)
    error('Does not support this option')
end
%% ---------- Plot --------------------------------------
if p.Results.new_fig; figure; hold on; end  %set(gcf,'Position',[1,41,1920,963]);
for img  = 1:num_example
    if p.Results.new_fig;subplot(p.Results.row, p.Results.col, img_ii);end
    hold on;grid off;axis off
    % ------ process reconstruction value ----------------------------
    if sum(abs(value(img,:)))>0
        tmp_value = value(img,:)';
        tmp_value(abs(tmp_value) < p.Results.thre*max(abs(tmp_value))) = 0;
        if p.Results.normalize
            tmp_value = tmp_value / max(abs(tmp_value)); % normalize
        end
    else
        tmp_value = zeros(size(pos,1),1);
    end
    % -------   define the source --------------------------------------
    % there is source location input
    if ~isempty(p.Results.source{1})
        if isstruct(p.Results.source{1})
            % provide new pos and tri for the source
            source_stru = p.Results.source{1};
            hpatch = patch ( 'vertices',source_stru.pos,'faces',source_stru.tri,'FaceVertexCData',ones(size(source_stru.pos,1),1)*[0 150 0]);
            set ( hpatch,'EdgeColor','none','FaceColor','interp','FaceLighting','phong','FaceAlpha',p.Results.SourceFaceAlpha,'DiffuseStrength',0.8 );
        else
            % use source index to plot the source
            % change the source patch a bit
            [azimuth,elevation,r] = cart2sph(pos(:,1),pos(:,2),pos(:,3));
            [x,y,z] = sph2cart(azimuth,elevation,r*1.05);
            source_pos = [x, y, z];
            hold on
            % if we have the same source for all the reconstruction input
            if length(p.Results.source) == 1
                ind = p.Results.source{1};
            else
                ind = p.Results.source{img};
            end
            % whether this is a point source 
            if length(ind) > 1
                pi = false(size(tri(:,1)));
                for i = 1:length(ind)
                    pi = pi | (tri(:,1)== ind(i)) | (tri(:,2) == ind(i)) | (tri(:,3) == ind(i));
                end
%                 pi = find(pi);
%                 for i=1:length(pi)
%                     if sum(ismember(tri(pi(i),:),ind)) < 2
%                         pi(i) = -1;
%                     end
%                 end
%                 pi(pi==-1) = [];
                hpatch = patch ( 'vertices',source_pos,'faces',tri(pi,:),'FaceVertexCData',ones(size(pos,1),1)*[0 150 0]);
                set ( hpatch,'EdgeColor','none','FaceColor','interp','FaceLighting','phong','FaceAlpha',p.Results.SourceFaceAlpha,'DiffuseStrength',0.8 );
            else
                scatter3(pos(ind,1),pos(ind,2),pos(ind,3),20,'green')
            end
        end
    end
    % -------   plot the cortex  ------------------------------------------
    if size(tmp_value,1) == size(pos,1)
        hpatch = patch ( 'vertices',pos,'faces',tri,'FaceVertexCData',tmp_value);
        set ( hpatch,'EdgeColor','none','FaceColor','interp','FaceLighting','phong','DiffuseStrength',1,'FaceAlpha',p.Results.FaceAlpha);
    else
        hpatch = patch ( 'vertices',pos,'faces',tri,'CData',tmp_value);
        set ( hpatch,'EdgeColor','none','FaceColor','flat','FaceLighting','phong','DiffuseStrength',1,'FaceAlpha',p.Results.FaceAlpha);
    end
    % -------   define the color map --------------------------------------
    if p.Results.neg  % Use [-1,1] colormap
        cmap1 = hot(38);
        cmap1 = cmap1(end:-1:1,:);
        cmap2 = jet(64);
        part1 = cmap1(1:31,:);
%         part2 = cmap2(1:30,:);
%         part2 = [part2;part1(1,:)];
        part2 = [0,0,0.5625;0,0,0.625;0,0,0.6875;0,0,0.75;0,0,0.8125;0,0,0.875;0,0,0.9375;0,0,1;0,0.0769230797886848,1;0,0.153846159577370,1;0,0.230769231915474,1;0,0.307692319154739,1;0,0.384615391492844,1;0,0.461538463830948,1;0,0.538461565971375,1;0,0.615384638309479,1;0,0.692307710647583,1;0,0.769230782985687,1;0,0.846153855323792,1;0,0.923076927661896,1;0,1,1;0.0625000000000000,1,0.937500000000000;0.125000000000000,1,0.875000000000000;0.187500000000000,1,0.812500000000000;0.250000000000000,1,0.750000000000000;0.312500000000000,1,0.687500000000000;0.375000000000000,1,0.625000000000000;0.531250000000000,1,0.718750000000000;0.687500000000000,1,0.812500000000000;0.843750000000000,1,0.906250000000000;1,1,1];
        mid_tran = gray(64);
        mid = mid_tran(57:58,:);
        cmap = [part2;mid;part1];
        caxis([-1 1])
    elseif p.Results.green  % for ground truth plot
        mid_tran = gray(64);
        cmap = mid_tran(57:end,:);
        cmap=[cmap;[0 1 0];[0 1 0]];
        caxis([0 1])
    else
        cmap1 = hot(38);  %32
        cmap1 = cmap1(end:-1:1,:);
        cmap2 = jet(64);
        part1 = cmap1(1:31,:);
        mid_tran = gray(64);
        mid = mid_tran(57:58,:);
        cmap = [mid;part1];
        caxis([0 1])
    end
    colormap(cmap)
    light('position',[3,3,1])
    light('position',[-3,-3,-1])
%     camlight HEADLIGHT;
%     camlight right;
%     camlight left;
    if size(p.Results.view,1) == 1
        view(p.Results.view);
    else
        view(p.Results.view(img,:));
    end

    % if the images fill the sub figure
    if img_ii == img_per_fig
        if p.Results.save
            fname = [p.Results.save, '/img' , int2str(img), '_', int2str(now*1e4)];
%             savefig(fname)
            saveas(gcf,[fname '.jpg'])
        end
        img_ii = 1;
        % there are more image to go 
        if img < num_example
            figure;
%             set(gcf,'Position',[1,41,1920,963]);
        end
    else
        img_ii = img_ii + 1;
    end
    if p.Results.titles{1}
        title(p.Results.titles{img})
    end
end

% save the last figure if not saved
if mod(num_example, img_per_fig) ~=0
    if p.Results.save
        fname = [p.Results.save, '/img' , int2str(img), '_', int2str(now*1e4)];
%         savefig(fname)
        saveas(gcf, [fname '.jpg'])
    end
end
end