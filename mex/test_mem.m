clear;
close all;

N = 10000;
data = randn(N,10);
data(1:N/2, 10) = 0;
data(N/2+1:N, 10) = 1;

% outputs: row wise matrixes
for ii = 1:10
    disp(ii);
    %[mfs_values, mfs_names] = metafeature(data);
    [mfs_values, mfs_names] = metafeature_mp(data);
    %[mfs_values, mfs_names] = metafeature_cu(data);
end
