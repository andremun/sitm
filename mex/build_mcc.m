%module load gcc/4.8.0 matlab/R2013a opencv/2.4.6.1
%matlab -nojvm < build_mcc.m

build_type = 0; % 0: normal, 1: openmp, 2: cuda

if build_type == 0
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -I../src -I/opt/sw/opencv-2.4.6.1/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -L/opt/sw/opencv-2.4.6.1/lib -lrt -lz -lopencv_core -lopencv_ml -cxx sitm.o Data.o -output sitm
 
elseif build_type == 1
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -DUSE_OpenMP CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -I/opt/sw/opencv-2.4.6.1/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -DUSE_OpenMP CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -L/opt/sw/opencv-2.4.6.1/lib -lrt -lz -lopencv_core -lopencv_ml -cxx sitm.o Data.o -output sitm_mp
   
end
