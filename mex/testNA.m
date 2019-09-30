clear;

data = [
    1,1.2,2,NaN,0;
    3,1.2,2,5,0;
    NaN,1.2,3,4,1;
    0,1.2,2,3,1;
    1,1.2,2,3,0;
    2,1.2,1,5,0
    ];

col_type = [0 0 1 1 2];

[mf_values, mf_names] = sitm(data, 2, col_type);