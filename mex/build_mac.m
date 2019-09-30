build_type = 0; % 0: normal, 1: openmp, 2: cuda

if build_type == 0
    mex -v -largeArrayDims -I../src -I/usr/include -I/usr/local/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims CXXLIBS="\$CXXLIBS -L/usr/local/lib/ -lopencv_core -lopencv_ml" -cxx sitm.o Data.o -o sitm

elseif build_type == 1
    mex -v -largeArrayDims -DUSE_OpenMP GXX= CXXFLAGS="\$CXXFLAGS -fopenmp" -I../src -I/usr/include -I/usr/local/include -cxx -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims -DUSE_OpenMP CXXFLAGS="\$CXXFLAGS -fopenmp" CXXLIBS="\$CXXLIBS -L/usr/local/lib/ -fopenmp -lopencv_core -lopencv_ml" -cxx sitm.o Data.o ...
        -o sitm_mp

elseif build_type == 2
    mexcuda -v -largeArrayDims -DUSE_CUDA -I../src -I/usr/include -I/usr/local/include -cxx -c ../src/cuda/dmat.cu ../src/Data.cpp sitm.cpp
    mexcuda -v -largeArrayDims -DUSE_CUDA CXXLIBS="\$CXXLIBS -L/usr/local/lib/ -lopencv_core -lopencv_ml" -cxx dmat.o sitm.o Data.o -o sitm_cu
end
