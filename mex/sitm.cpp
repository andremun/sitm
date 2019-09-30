#include <matrix.h>
#include <mex.h>
#include <time.h>
#include <math.h>
#ifdef USE_OpenMP
#include <omp.h>
#endif

//#define PRINT_DEBUG

#define BUILD_ID 1001 //28-09-2016

#ifdef PRINT_DEBUG
#include <fstream>
#endif

using namespace std;

#include "../src/Data.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#ifdef USE_OpenMP
  mexPrintf("SITM with openmp (build %d)...\n", BUILD_ID);
#else
  mexPrintf("SITM (build %d)...\n", BUILD_ID);
#endif

  size_t nrows, ncols;
  double* matlab_data;    // row wise
  int na_option;
  double* matlab_col_type = NULL;
  double* matlab_mask = NULL;
 
  if(nrhs < 2 || nrhs > 4) {
    mexPrintf("Error: Incorrect input arguments\n");
    mexPrintf("Usage: sitm(data, na_option) or sitm(data, na_option, col_type) or sitm(data, na_option, col_type, mask).\n");
    return;
  }
  const mwSize *dims = mxGetDimensions(prhs[0]);
  nrows = dims[0]; ncols = dims[1];
#ifdef PRINT_DEBUG
  mexPrintf("nrows: %d, ncols: %d\n", nrows, ncols);
#endif
  matlab_data = mxGetPr(prhs[0]);
  na_option = (int)mxGetScalar(prhs[1]);
  if(na_option < 0 || na_option > 2) {
    mexPrintf("Error: na_option must be in {0: NA_KEEP, 1: NA_REMOVE, 2: NA_ESTIMATE}\n");
      return;
  }

  if(nrhs == 3 || nrhs == 4) {
    matlab_col_type = mxGetPr(prhs[2]);
    if(mxGetN(prhs[2]) != ncols) {
      mexPrintf("Error: col_type must have %d elements\n", ncols);
      return;
    }
    for(int i=0; i < ncols-1; i++) {
      if(matlab_col_type[i] != 0 && matlab_col_type[i] != 1) {
        mexPrintf("Error: data att type must be 0 (numeric) or 1 (nominal)\n");
        return;
      }
    }
    if(matlab_col_type[ncols-1] != 2) {
      mexPrintf("Error: att type of the last column must be 2 (CLASS)\n");
      return;
    }
  }

  if(nrhs == 4) {
    matlab_mask = mxGetPr(prhs[3]);
    if(mxGetN(prhs[3]) != NUM_FEATURES) {
      mexPrintf("Error: mask must have %d elements\n", NUM_FEATURES);
      return;
    }
  }

  int* col_type = new int[ncols];
  bool* mask = new bool[NUM_FEATURES];
  for(int c = 0; c < ncols-1; c++) {
    if(matlab_col_type)
      col_type[c] = (int)matlab_col_type[c];
    else
      col_type[c] = NUMERICAL;
  }
  col_type[ncols-1] = CLASS;

  int count = 0;
  for(int i=0; i < NUM_FEATURES; i++) {
    if(matlab_mask)
      mask[i] = (int)matlab_mask[i] != 0;
    else
      mask[i] = true;
    if(mask[i])
      count++;
  }

  Data* d = new Data(na_option);

  plhs[0] = mxCreateDoubleMatrix(count, 1, mxREAL); 
  double *mfs_values = mxGetPr(plhs[0]);
  int f_ind = 0;
  for(int i=0; i < NUM_FEATURES; i++) {
    if(mask[i])
      mfs_values[f_ind++] = -1;
  }
 
  f_ind = 0;
  plhs[1] = mxCreateCellMatrix(count, 1);
  for(int i=0; i < NUM_FEATURES; i++) {
    if(mask[i]) {
      mxSetCell(plhs[1], f_ind, mxCreateString(d->getMFStringByID(i).c_str()));
      f_ind++;
    }
  }
  
  if(d->loadDataFromMatlab(matlab_data, nrows, ncols, col_type, mask)) {
    mexPrintf("Error: fail to load data or process NA or # of classes < 2\n");
    return;
  }
  
#ifdef PRINT_DEBUG
  ofstream ofs;
  string outfilename = "data1_result.txt";
  ofs.open(outfilename, ofstream::out);
  d->print(true, ofs);
#endif

  unsigned int t = d->getTime();
  d->calMFs();
 
#ifdef PRINT_DEBUG
  d->printMFs(ofs);
  if(ofs)
    ofs.close();
#endif

  f_ind = 0;
  for(int i=0; i < NUM_FEATURES; i++) {
    if((int)mask[i])
      mfs_values[f_ind++] = d->getMFValueByID(i);
  }
  
  delete d;
  delete []col_type;
  delete []mask;

  mexPrintf("SITM - processing time: %d\n", d->getTime() - t);
}


