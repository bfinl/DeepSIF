function [rm, names] = fs_get_rm(subpath, annot_name)
% Combine rh and lh .annot file
% Input subpath:subject path
%       annot_name:annotation file name
[~, label1, colortable1] = freesurfer_read_annotation([subpath '\lh.' annot_name '.annot']);
[~, label2, colortable2] = freesurfer_read_annotation([subpath '\rh.' annot_name '.annot']);
n1 = length(colortable1.struct_names);
n2 = length(colortable2.struct_names);
%% sort colortable to find the new rm
% colortable1.struct_names{cellfun('isempty',colortable1.struct_names)} = ' ';
% colortable2.struct_names{cellfun('isempty',colortable2.struct_names)} = ' ';
if strcmp(annot_name, 'nmm_994')
    names = [colortable1.struct_names;colortable2.struct_names];
    names_tmp = names;
    names_tmp{strcmp(names, '994')}='7';
    names_tmp{strcmp(names, '997')}='325';
    names_tmp{strcmp(names, '996')}='921';
    names_tmp{strcmp(names, '995')}='949';
%     [~,new_ind] = natsort(names_tmp); 
    new_ind = cellfun(@str2num,names_tmp)+1;
    names(new_ind) = names;
else%if %strcmp(annot_name, 'laus500')
    names = [colortable1.struct_names;colortable2.struct_names];
    new_ind = [1:n1 1 (n1+1):(n1+n2-1)];
end

%% map to rm
mapped_label1 = zeros(size(label1));
for k = 1:n1
    mapped_label1(label1==colortable1.table(k,5))=new_ind(k)-1;
end
mapped_label2 = zeros(size(label2));
for k = 1:n2
    mapped_label2(label2==colortable2.table(k,5))=new_ind(n1+k)-1;
end

rm = [mapped_label1' mapped_label2'];

end