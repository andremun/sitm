%module load matlab/r2015b
%matlab -nojvm < build_monarch.m

build_type = 0; % 0: normal, 1: openmp

if build_type == 0
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -I../src -I/home/toand/git/opencv-2.4.13/build/install/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -L/home/toand/git/opencv-2.4.13/build/install/lib -lrt -lz -lopencv_core -lopencv_ml -cxx sitm.o Data.o -output sitm
 
elseif build_type == 1
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -DUSE_OpenMP CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -I../src -I/home/toand/git/opencv-2.4.13/build/install/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -DUSE_OpenMP CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -L/home/toand/git/opencv-2.4.13/build/install/lib -lrt -lz -lopencv_core -lopencv_ml -cxx sitm.o Data.o -output sitm_mp
   
end
