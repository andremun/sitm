all:
	nvcc -c dmat.cu
	g++-5 -o test test.cpp dmat.o -L/Developer/NVIDIA/CUDA-7.5/lib -lcudart

clean: rm -rf *.o