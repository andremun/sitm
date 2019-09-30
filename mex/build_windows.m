% Change C:/opencv/build to your opencv
% Tested with: Windows 7, CUDA 7.0.27, MSVC 2010 Pro

build_type = 0; % 0: normal, 1: openmp, 2: cuda

if build_type == 0
    mex -v -largeArrayDims -I../src -IC:/opencv/2.4.13/build/include -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims sitm.obj Data.obj C:/opencv/2.4.13/build/x64/vc12/lib/opencv_core2413.lib ...
        C:/opencv/2.4.13/build/x64/vc12/lib/opencv_ml2413.lib -output sitm

elseif build_type == 1
    mex -v -largeArrayDims -DUSE_OpenMP COMPFLAGS="/openmp $COMPFLAGS" -I../src -IC:/opencv/2.4.13/build/include -c ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims -DUSE_OpenMP COMPFLAGS="/openmp $COMPFLAGS" sitm.obj Data.obj ...
        C:/opencv/2.4.13/build/x64/vc12/lib/opencv_core2413.lib C:/opencv/2.4.13/build/x64/vc12/lib/opencv_ml2413.lib -output sitm_mp

elseif build_type == 2
    system(sprintf('nvcc -I"%s/extern/include" --compiler-options="/MD" --cuda "../src/cuda/dmat.cu" --output-file "dmat.cpp"', matlabroot));
    mex -v -largeArrayDims -DUSE_CUDA -I../src -I../src/cuda -IC:/opencv/2.4.13/build/include -c dmat.cpp ../src/Data.cpp sitm.cpp
    mex -v -largeArrayDims -DUSE_CUDA dmat.obj sitm.obj Data.obj ...
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cudart_static.lib' ...
            C:/opencv/2.4.13/build/x64/vc12/lib/opencv_core2413.lib C:/opencv/2.4.13/build/x64/vc12/lib/opencv_ml2413.lib -output sitm_cu
end