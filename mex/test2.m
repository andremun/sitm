clear;

load('spambase.mat');

data = cell2mat(cellfun(@(x) double(x),struct2cell(data),'UniformOutput',false)');
var_type(end) = 2; % Make this a class attribute
data(data==-2.147483648000000e+09) = NaN;

[mfs_values02, mfs_names02] = sitm(data, 0, var_type');