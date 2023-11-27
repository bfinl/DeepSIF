function [pos, tri, ind] = fs_get_surf(fpath, varargin)
% Input: the file path of the pial file
% 'fname': the function will read fpath+'r/l'+fname
p = inputParser;
addRequired(p,'fpath',@isstr);
addParameter(p,'fname','h.pial', @isstr);
addParameter(p,'half','', @isstr);
parse(p, fpath, varargin{:})
fpath = p.Results.fpath;
fname = p.Results.fname;
[vertices1, faces1] = freesurfer_read_surf([fpath '\r' fname ]);
[vertices2, faces2] = freesurfer_read_surf([fpath '\l' fname ]);
if strcmp(p.Results.half,'left')
    pos = vertices2;
    tri = faces2;
    ind = 1:size(pos,1);
elseif strcmp(p.Results.half,'right')
    pos = vertices1;
    tri = faces1;
    ind = (size(vertices2,1)+1):(size(pos,1)+size(vertices2,1));
else
    pos = [vertices2;vertices1];
    tri = [faces2;faces1+size(vertices2,1)];
    ind = 0; 
end
   
end