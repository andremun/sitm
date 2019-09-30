%MATLAB 2015b, gcc4.7.4
%Need to run massive_prepare_run_matlab first to have correct modules

build_type = 1; % 0: normal, 1: openmp, 2: cuda

if build_type == 0
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -I../src -I/home/toand/dev/opencv-2.4.12/build/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -L/home/toand/dev/opencv-2.4.12/build/lib -lrt -lz -lopencv_core -lopencv_ml -cxx sitm.o Data.o -output sitm
 
elseif build_type == 1
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -DUSE_OpenMP CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -I../src -I/home/toand/dev/opencv-2.4.12/build/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O2" -DUSE_OpenMP CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -L/home/toand/dev/opencv-2.4.12/build/lib -lrt -lz -lopencv_core -lopencv_ml -cxx sitm.o Data.o -output sitm_mp
   
elseif build_type == 2
    mexcuda -v -largeArrayDims -DUSE_CUDA -I../src -I../src/cuda -I/home/toand/dev/opencv-2.4.12/build/include -cxx -c ../src/cuda/dmat.cu ../src/Data.cpp sitm.cpp
    mexcuda -v -largeArrayDims -DUSE_CUDA -L/home/toand/dev/opencv-2.4.12/build/lib -lrt -lz -lopencv_core -lopencv_ml -cxx dmat.o sitm.o Data.o -output sitm_cu
end
