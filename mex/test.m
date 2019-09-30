clear;
close all;

test_case = 1;  % 1: current_1 (958 x 10, 2 classes) 
                % 2: IRIS (150 x 5, 3 classes) 
                % 3: abalone: 4177 x 9, 3 classes, first col is nominal

test_data = load('test_data.mat');

% the last colume of data is CLASS
data = test_data.test{test_case}.data;
col_type = test_data.test{test_case}.col_type;
na_option = 2;

%MFs: 29 in total
%"nInst", "nAtt", "nClasses", "pNom", "classProbSD",
%"pNAinst", "Fract1", "ClSD.Mean", "ClCV.Max", "ClCV.Mean",
%"ClSkew.Min", "ClSkew.Mean", "entAttN.Min", "entAttN.Max", "entClassN",
%"jEntAC.Mean", "jEntAC.SD", "jEntAC.Skew", "mutInfoAC.Mean", "mutInfoAC.SD",
%"NSratio", "DN", "WN", "conceptVariation.Mean", "conceptVariation.StdDev",
%"conceptVariation.Kurtosis", "weightedDist.Mean",
%"weightedDist.StdDev", "weightedDist.Skewness"
sitm_mask = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0]; %0: disabled (no need to calculate)

% outputs: row wise matrixes
[mfs_values01, mfs_names01] = sitm(data, na_option);
[mfs_values02, mfs_names02] = sitm(data, na_option, col_type);
[mfs_values03, mfs_names03] = sitm(data, na_option, col_type, sitm_mask);

[mfs_values11, mfs_names11] = sitm_mp(data, na_option);
[mfs_values12, mfs_names12] = sitm_mp(data, na_option, col_type);
[mfs_values13, mfs_names13] = sitm_mp(data, na_option, col_type, sitm_mask);