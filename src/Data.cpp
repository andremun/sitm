#include "Data.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <set>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <iomanip>

#ifdef MATLAB_DEBUG
#include <matrix.h>
#include <mex.h>
#endif

#ifdef USE_OpenMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "dmat.h"
#endif

#include <opencv2/ml/ml.hpp>

using namespace std;

#if _WIN32
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64
#include <ctime>

#ifndef NAN
    static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
    #define NAN (*(const float *) __nan)
#endif

#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define fpu_error(x) (isinf(x) || isnan(x))

int gettimeofday(struct timeval * tp, struct timezone * tzp) {
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#else
#include <sys/time.h>
#define isnan(x) std::isnan(x)
#define isinf(x) std::isinf(x)
#define fpu_error(x) (std::isinf(x) || std::isnan(x))
#endif

static float myroundf(float x) {
   return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
}

const string Data::_feature_names[NUM_FEATURES] = {"nInst", "nAtt", "nClasses", "pNom", "classProbSD",
                   "pNAinst", "Fract1", "ClSD.Mean", "ClCV.Max", "ClCV.Mean",
                   "ClSkew.Min", "ClSkew.Mean", "entAttN.Min", "entAttN.Max", "entClassN",
                   "jEntAC.Mean", "jEntAC.SD", "jEntAC.Skew", "mutInfoAC.Mean", "mutInfoAC.SD",
                   "NSratio", "DN", "WN", "conceptVariation.Mean", "conceptVariation.StdDev",
                   "conceptVariation.Kurtosis", "weightedDist.Mean",
                   "weightedDist.StdDev", "weightedDist.Skewness"}; // 29 MFs

Data::Data(int na_option) : _n_instances(0), _n_attributes(0), _na_option(na_option), 
                             _na_row_mask(NULL), _na_col_mask(NULL) {
	_header.clear();
    _class_value.clear();
    _class_value_valid.clear();
}

Data::~Data()
{
    _data_mat.release();
    _numeric_mat.release();
    _discretize_mat.release();
    _numeric_mat_valid.release();
    if(_na_row_mask) 
        delete [] _na_row_mask;
    if(_na_col_mask)
        delete [] _na_col_mask;
}

unsigned int Data::getTime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (tp.tv_sec * 1000 + tp.tv_usec / 1000);
}

int Data::loadDataFromArffFile(const char* filename, const bool* mask) {
	ifstream infile(filename);
    if(!infile) {
        cout << "Error: cannot open file " << filename << " to read" << endl;
        return 1;
    }
  
	string line;
	
    _n_instances = 0;

    //parse header and count dimensions
    while (getline(infile, line)) {

		if (line.length() == 0)
			continue;

        if(line[0] == '@') { //attribute or class
            stringstream  ss(line);
            string  token;
            ss >> token;
            if(token.compare("@relation") == 0 || token.compare("@data") == 0)
                continue;
            ss >> token; // name
            bool isclass = false;
            if(token.compare("class") == 0)
                isclass = true;

            ss >> token; // data
            Attribute att;
            if(token.compare("numeric") == 0) {
                att.type = NUMERICAL;
            }
            else {
                att.type = isclass ? CLASS : NOMINAL;
                // set str_int
                token.erase (0,1);                      // remove {
                token.erase (token.length()-1, 1);      // remove }
                stringstream values(token);
                string v_str;
                int v_value = 1;                        // noninal attribte starts from 1 (1, 2, 3, ...) as in R (?)
                while (getline(values, v_str, ',')) {
                    if(v_str[0]!='\'')
                        att.str_int[v_str] = atoi(v_str.c_str());
                    else
                        att.str_int[v_str] = v_value++;
                }
                att.num_category = att.str_int.size();
            }
            _header.push_back(att);
        }
        else {
            _n_instances++;
        }
    }

    _n_attributes = _header.size();

	// read matrixes
	infile.clear();
	infile.seekg(0);

    _data_mat.create(_n_instances,_n_attributes, CV_64F);
    _na_row_mask = new bool[_n_instances]();
    _na_col_mask = new bool[_n_attributes]();
	
	int line_index = 0;
    while (getline(infile, line)) {

		if (line.length() == 0)
			continue;

        if(line[0] == '@')
            continue;

        stringstream ss(line);
        string token;
        int ind = 0;
        while (getline(ss, token, ',')) {
            if(token.compare("?") == 0) {
                _data_mat.at<double>(line_index, ind) = NAN;
                _has_na = true;
                _na_row_mask[line_index] = true;
                _na_col_mask[ind] = true;
            }
            else {
                if (_header[ind].type == NUMERICAL)
                    _data_mat.at<double>(line_index, ind) = atof(token.c_str());
                else    
                    _data_mat.at<double>(line_index, ind) = _header[ind].str_int[token];
            }
            ind++;
        }
        line_index++;
	}

    if(initData()) {
        cout << "Error: fail to load data or process NA or # of class < 2" << endl;
        return 1;
    }

    // feature mask
    for(int i=0; i < NUM_FEATURES; i++) {
        _MFs[_feature_names[i]] = -1;
        _MFs_mask[_feature_names[i]] = mask[i] != 0;
    }

	return 0;
}

// Note: MATLAB uses col-wise matrix
int Data::loadDataFromMatlab(const double* data, const int rows, const int cols, const int* col_type, const bool* mask) {
    _n_instances = rows;
    _n_attributes = cols;

    //header
    for(int i=0; i < cols-1; i++) {
        Attribute att;
        if(col_type) {
            att.type = (int)col_type[i];
            att.num_category = 1;
            for(int r=0; r < rows; r++)
                if(!isnan(data[r+i*rows]) && data[r+i*rows] > att.num_category)
                    att.num_category = data[r+i*rows];
        }
        else
            att.type = NUMERICAL;
        _header.push_back(att);
    }
    Attribute att;
    att.type = CLASS;
    _header.push_back(att);

    //data
    _data_mat.create(rows, cols, CV_64F);
    _na_row_mask = new bool[rows]();
    _na_col_mask = new bool[cols]();
    for(int row=0; row < rows; row++)
        for(int col=0; col < cols; col++) {
            if(isnan(data[row+col*rows])) {
                _data_mat.at<double>(row, col) = NAN;
                _has_na = true;
                _na_row_mask[row] = true;
                _na_col_mask[col] = true;
            }
            else
                _data_mat.at<double>(row, col) = data[row+col*rows];
        }

    if(initData())
        return 1;

    // feature mask
    for(int i=0; i < NUM_FEATURES; i++) {
        _MFs[_feature_names[i]] = -1;
        _MFs_mask[_feature_names[i]] = mask[i] != 0;
    }

    return 0;
}

int Data::initData() {

    switch (_na_option) {
        case NA_KEEP:
            break;
        case NA_REMOVE:
            removeNA();
            break;
        case NA_ESTIMATE:
            estimateNA();
            break;
        default:
            break;
    };

    if(_n_instances < 2)
        return 1;
    
    // class values and labels (unique values)
    for(int i=0; i < _n_instances; i++) {
        _class_value.push_back(_data_mat.at<double>(i,_n_attributes-1));
        _class_label.insert(_data_mat.at<double>(i,_n_attributes-1));
        if(!_na_row_mask[i])
            _class_value_valid.push_back(_data_mat.at<double>(i,_n_attributes-1));
    }

    if(_class_label.size() < 2)
        return 1;

    // class weight
    for(set<int>::iterator it=_class_label.begin(); it!= _class_label.end(); it++) {
        int c = *it;
        _class_weight[c] = (double)std::count (_class_value.begin(), _class_value.end(), c) / _n_instances;
    }

    // all-numeric mat (data mat without class (last) col)
    _numeric_mat.create(_n_instances, _n_attributes-1, CV_64F);
    int ind = 0;
    for(int i = 0; i < _n_attributes-1; i++)
        _data_mat.col(i).copyTo(_numeric_mat.col(ind++));

    // _numeric_mat_valid
    if(_na_option != NA_REMOVE) {
        int num_valid_rows = 0;
        for(int i=0; i < _n_instances; i++)
            if(!_na_row_mask[i])
                num_valid_rows++;
        _numeric_mat_valid.create(num_valid_rows, _numeric_mat.cols, CV_64F);
        ind = 0;
        for(int i=0; i < _n_instances; i++) {
            if(!_na_row_mask[i])
                _numeric_mat.row(i).copyTo(_numeric_mat_valid.row(ind++));
        }    
    }
    
    // calculate discratize matrix with EquaFreq algorithm
    _discretize_mat = discretizeEquaFreq(_numeric_mat);

    return 0;
}

void Data::removeNA() {
    if(!_has_na) {
        _data_mat.copyTo(_numeric_mat_valid);
        return;
    }

    int num_nonna_attributes = 0;
    for(int i=0; i < _n_instances; i++)
        if(!_na_row_mask[i])
            num_nonna_attributes++;

    cv::Mat temp;
    temp.create(num_nonna_attributes, _n_attributes, CV_64F);

    int ind = 0;
    for(int i=0; i < _n_instances; i++) {
        if(!_na_row_mask[i])
            _data_mat.row(i).copyTo(temp.row(ind++));
    }    

    _data_mat = temp;
    _n_instances = _data_mat.rows;
    _data_mat.copyTo(_numeric_mat_valid);
}

double Data::estimateColValue(const vector<double>& valid, int col) {
    double v;

    if(_header[col].type == NUMERICAL) {
        double sum = std::accumulate(valid.begin(), valid.end(), 0.0);
        v = sum / valid.size();

    } else { // NOMINAL
        map<int, int> freq;
        for(int i=0; i < valid.size(); i++) {
            int evalue = (int)valid[i];
            freq[evalue]++;
            /*
            map<int, int>::iterator it= freq.find(evalue);
            if( it != freq.end() )
                freq[evalue]++;
            else
                freq[evalue] = 1;
            */
        }
        int count = 0;
        for(map<int, int>::iterator it= freq.begin(); it != freq.end(); it++) {
            if(it->second > count) {
                count = it->second;
                v = it->first;
            }
        }
    }

    return v;
}

void Data::estimateNA() {
    if(!_has_na)
        return;

    vector<int> class_value;  
    set<int> class_label;
    for(int row=0; row < _n_instances; row++) {
        class_value.push_back(_data_mat.at<double>(row, _n_attributes-1));
        class_label.insert(_data_mat.at<double>(row, _n_attributes-1));
    }

    for(int col = 0; col < _n_attributes-1; col++) {
        if(!_na_col_mask[col])
            continue;

        vector<double> valid;

        for(set<int>::iterator it=class_label.begin(); it!= class_label.end(); it++) {
            int c = *it;
            valid.clear();
            for(int row = 0; row < _n_instances; row++) {
                if (class_value[row] == c && !isnan(_data_mat.at<double>(row, col)))
                    valid.push_back(_data_mat.at<double>(row, col));
            }
            if(valid.size() == 0)
                continue;

            double estvalue = estimateColValue(valid, col);
            for(int row = 0; row < _n_instances; row++) {
                if (class_value[row] == c && isnan(_data_mat.at<double>(row, col)))
                    _data_mat.at<double>(row, col) = estvalue;
            }
        }

        // If after checking all the variables there are still nans, use a
        // global inputation method
        vector<int> naind;
        for(int row=0; row < _n_instances; row++) {
            if (isnan(_data_mat.at<double>(row, col)))
                naind.push_back(row);
        }
        if(naind.size() == 0)
            continue;
        valid.clear();
        for(int row = 0; row < _n_instances; row++) {
            if (!isnan(_data_mat.at<double>(row, col)))
                valid.push_back(_data_mat.at<double>(row, col));
        }
        double estvalue = estimateColValue(valid, col);
        for(int i=0; i < naind.size(); i++) 
            _data_mat.at<double>(naind[i], col) = estvalue;
    }
}

void Data::printMat(const char* name, const cv::Mat mat) {
    cout << name << endl;
    for(int i=0; i < mat.rows; i++) {
        cout << setw(4) << i+1 << "  ";
        for(int j=0; j < mat.cols; j++) {
            if(isnan(mat.at<double>(i, j)))
                cout << "? ";
            else
                cout << mat.at<double>(i, j) << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void Data::printMat(const char* name, const cv::Mat mat, std::ofstream& ofs) {
    cout << name << endl;
    for(int i=0; i < mat.rows; i++) {
        cout << setw(5) << i+1 << "  ";
        for(int j=0; j < mat.cols; j++) {
            if(isnan(mat.at<double>(i, j)))
                cout << "? ";
            else
                cout << mat.at<double>(i, j) << " ";
        }
        cout << endl;
    }
    cout << endl;
    if(ofs) {
        ofs << name << endl;
        for(int i=0; i < mat.rows; i++) {
            for(int j=0; j < mat.cols; j++) {
                ofs << mat.at<double>(i, j) << " ";
            }
            ofs << endl;
        }
        ofs << endl;
    }
}

void Data::print(bool print_matrix, ofstream& ofs) {
    cout << "====== info ======" << endl;
#ifdef USE_OpenMP
    cout << "(Built with OpenMP)" << endl;
#endif
#ifdef USE_CUDA
    cout << "(Built with CUDA)" << endl;
#endif

    cout << "num rows: " << _n_instances << endl;
    cout << "num attributes (including class): " << _n_attributes << endl;
   
    cout << "datatype: ";
    for (int i = 0; i < _header.size(); i++)
        cout << _header[i].type << " ";
    cout << endl;

    if(ofs) {
        ofs << "====== info ======" << endl;
        ofs << "num rows: " << _n_instances << endl;
        ofs << "num attributes (including class): " << _n_attributes << endl;
        ofs << "col type: ";
        for (int i = 0; i < _header.size(); i++)
            ofs << _header[i].type << " ";
        ofs << endl;
    }

    if (print_matrix) {
        cout << "class: ";
        for (int i = 0; i < _class_value.size(); i++)
            cout << _class_value[i] << " ";
        cout << endl;

        printMat("data matrix:", _data_mat, ofs);

        printMat("numeric matrix:", _numeric_mat, ofs);

        printMat("discretized matrix:", _discretize_mat, ofs);

        printMat("numeric_mat_valid matrix:", _numeric_mat_valid, ofs);

    }

    cout << "==================" << endl << endl;
    if(ofs)
        ofs << "==================" << endl << endl;
}

double Data::getMFbyName(string mf_name) {
    if (_MFs.find(mf_name) != _MFs.end())
        return _MFs[mf_name];
    return 0;
}

void Data::printMFs(ofstream& ofs) {
    cout << "====== result ======" << endl;
    for (int i=0; i < NUM_FEATURES; i++) {
        if(_MFs_mask[_feature_names[i]])
            cout << _feature_names[i] << ": " << _MFs[_feature_names[i]] << endl;
    }
    cout << "====================" << endl;

    if(ofs) {
        ofs << "====== result ======" << endl;
        for (int i=0; i < NUM_FEATURES; i++) {
            if(_MFs_mask[_feature_names[i]])
                ofs << _feature_names[i] << ": " << _MFs[_feature_names[i]] << endl;
        }
        ofs << "====================" << endl;
    }
}

// calculate discreteized matrix using Equa Freq (ref: discretize function in R infotheo package)
cv::Mat Data::discretizeEquaFreq(const cv::Mat data) {
    int nbins = (int)pow(data.rows, 1.0 / 3.0);
    if (nbins < 1)
        nbins = 1;
    return discretizeEquaFreq(data, nbins);
}

cv::Mat Data::discretizeEquaFreq(const cv::Mat data, const int nbins) {
    vector<double>spl;
    spl.resize(nbins);

    const double epsilon = 0.01;

    cv::Mat res = cv::Mat(data.rows, data.cols, CV_64F, cv::Scalar(0.0));

    for (int v = 0; v < data.cols; ++v) {

        if(_header[v].type == NOMINAL) {
            data.col(v).copyTo(res.col(v));
            continue;
        }

        // get column v
        vector<double> col;
        for(int r = 0; r < data.rows; ++r) {
            if(!isnan(data.at<double>(r,v)))
                col.push_back(data.at<double>(r, v));
        }
        std::sort(col.begin(), col.end()); // ascending
        int N = col.size();
        
        int freq = N / nbins, mod = N % nbins;
        int splitpoint = freq - 1;
        for (int i = 0; i < nbins-1; ++i) {

            if (mod > 0) {
                spl[i] = col[splitpoint + 1];
                mod--;
            }
            else {
                spl[i] = col[splitpoint];
            }

            splitpoint += freq;
        }

        spl[nbins - 1] = col[N - 1] + epsilon;

        for (int s = 0; s < data.rows; ++s)
        {
            if (!isnan(data.at<double>(s,v))) {
                int bin = -1;
                for (int k = 0; bin == -1 && k<nbins; ++k)
                    if (data.at<double>(s,v) <= spl[k])
                        bin = k;
                res.at<double>(s, v) = bin + 1;
            }
            else
                res.at<double>(s, v) = NAN;
        }
    }

    return res;
}

// col_mat: Nx1 matrix (column of a matrix
int Data::calStats(const cv::Mat col_mat, double &mean, double &std, double &skew, double &kurtosis) {
    int N = col_mat.rows;
    double sum = 0; int count = 0;
    for(int i=0; i < N; i++)
        if(!isnan(col_mat.at<double>(i, 0))) {
            sum += col_mat.at<double>(i, 0);
            count++;
        }
    mean = sum / count;
    
    double std_temp = 0, skew_temp = 0, kurtosis_temp = 0;
    for(int i=0; i < N; i++) {
        if(!isnan(col_mat.at<double>(i, 0))) {
            double v = col_mat.at<double>(i, 0) - mean;
            double v2 = v*v;
            std_temp += v2;
            skew_temp += v2*(col_mat.at<double>(i, 0) - mean);
            kurtosis_temp += v2*v2;
        }    
    }

    if(count < 1) {
        cout << "Warning: number of (valid) samples is less than 2! Cannot calculate std." << endl;
        std = 0;
    }
    else
        std = sqrt(std_temp / (count-1));

    skew_temp /= count;
    double std_n_3 = pow(sqrt(std_temp / count), 3);
    skew = skew_temp / std_n_3;

    //kurtosis
    kurtosis = count * kurtosis_temp / (std_temp*std_temp);

    return 0;
}

cv::Mat Data::calDistanceMatrix(const cv::Mat mat, const cv::Mat UL) {
    cv::Mat d_mat;
    int N = mat.rows;
    d_mat.create(N, N, CV_64F);
#ifdef USE_OpenMP
#pragma omp parallel for
#endif
    for(int i=0; i < N; i++) {
        for(int j=0; j < N; j++) {
            if (i == j)
                d_mat.at<double>(i, j) = 0;
            else if (i < j) {
                float dis = 0;
                for(int k=0; k < mat.cols; k++) {
                    if (UL.at<double>(0, k) != 0) {
                        float diff = (mat.at<double>(i, k) - mat.at<double>(j, k)) / UL.at<double>(0, k);
                        dis += diff*diff; 
                    }                     
                }
                dis = sqrt(dis);
                d_mat.at<double>(i, j) = dis;
                d_mat.at<double>(j, i) = dis;
            }
        }
    }
    return d_mat;
}

int Data::getNumEnabledFeatures() {
    int num = 0;
    for (int i=0; i < NUM_FEATURES; i++)
        if(_MFs_mask[_feature_names[i]])
            num++;
    return num;
}

bool Data::isFeatureEnabled(int i) {
    if(i > NUM_FEATURES)
        return false;
    return _MFs_mask[_feature_names[i]];
}

double Data::getMFValueByID(int i) {

    if(i > NUM_FEATURES || !_MFs_mask[_feature_names[i]])
        return -1;
    return _MFs[_feature_names[i]];
}

string Data::getMFStringByID(int i) {
    if(i > NUM_FEATURES)
        return "";
    return _feature_names[i];
}

// == calculate MFs ==
int Data::calMFs() {
    unsigned int t;

    // simple MFs
#ifdef PRINT_DEBUG
    t = getTime();
#endif
    calNumInstances();          //nInst
    calNumAttributes();         //nAtt
    calNumClasses();            //nClasses
    calNominalProportion();     //pNom
    calClassProbability();      //classProbSD
    calNaProportion();          //pNAinst
    calFract1();                //Fract1
    calClassSD_CV_Skew();       //ClSD.Mean, ClCV.Max, ClCV.Mean, ClSkew.Min, ClSkew.Mean
#ifdef PRINT_DEBUG
    cout << "simple MFs time: " << getTime() - t << endl;
#endif

	// information theoretic MFs
#ifdef PRINT_DEBUG
    t = getTime();
#endif
	calNomalizedEntropy();		//entAttN.Min, entAttN.Max
	calClassEntropy();			//entClassN
	calJointEntropy();			//jEntAC.Mean, jEntAC.SD, jEntAC.Skew
	calMutualInfo();			//mutInfoAC.Mean, mutInfoAC.SD
	calNoiseSignalRatio();		//NSratio
#ifdef PRINT_DEBUG
    cout << "information theoretic MFs time: " << getTime() - t << endl;
#endif

    // landmarking MFs
#ifdef PRINT_DEBUG
    t = getTime();
#endif
    calLandmarking();           //DN, WN
#ifdef PRINT_DEBUG
    cout << "calLandmarking time: " << getTime() - t << endl;
#endif

    // concept characterisation
#ifdef PRINT_DEBUG
    t = getTime();
#endif
    calConceptChar();           //conceptVariation.Mean, conceptVariation.StdDev
                                //conceptVariation.Kurtosis, weightedDist.Mean
                                //weightedDist.StdDev, weightedDist.Skewness
#ifdef PRINT_DEBUG
    cout << "calConceptChar time: " << getTime() - t << endl;
#endif

	return 0;
}

// simple MFs
int Data::calNumInstances() {
    _MFs["nInst"] = _n_instances;
    return 0;
}

int Data::calNumAttributes() {
    _MFs["nAtt"] = _numeric_mat.cols;
    return 0;
}

int Data::calNumClasses() {
    _MFs["nClasses"] = _class_label.size();
    return 0;
}

int Data::calNominalProportion() {
    int n_nominal = 0;
    for(int i=0; i < _header.size(); i++)
        if(_header[i].type == NOMINAL)
            n_nominal++;
    _MFs["pNom"] = (double)n_nominal / _numeric_mat.cols;
    return 0;
}

int Data::calClassProbability() {
    // count number of elements in classes
    vector<double> class_prob;
    for(set<int>::iterator it=_class_label.begin(); it!= _class_label.end(); it++) {
        int c = *it;
        double cp = (double)std::count (_class_value.begin(), _class_value.end(), c) / _n_instances;
        class_prob.push_back(cp);
    }

    if(class_prob.size() == 0) {
        cout << "Error: class_prob.size = 0" << endl;
        return 1;
    }

    //double sum = std::accumulate(class_prob.begin(), class_prob.end(), 0.0);
    //double mean = sum / class_prob.size();
    double mean = 1.0 / class_prob.size();
    double variance = 0;
    for(int i=0; i < class_prob.size(); i++)
        variance += (class_prob[i]-mean)*(class_prob[i]-mean);
    _MFs["classProbSD"] = std::sqrt(variance / (class_prob.size()-1));

    return 0;
}

int Data::calNaProportion() {
    if(_na_option != NA_KEEP || !_has_na )
        _MFs["pNAinst"] = 0;
    else {
        double na_instance_count = 0;
        for(int i=0; i < _n_instances; i++)
            if(_na_row_mask[i])
                na_instance_count++;
        _MFs["pNAinst"] = na_instance_count / _n_instances;
    }
    return 0;
}

int Data::calFract1() {
    if (_numeric_mat_valid.rows < 1) {
		cout << "Warning: (calFract1) number of valid rows <= 1" << endl;
		return 1;
	}

    cv::Mat col_mean;
    cv::reduce(_numeric_mat_valid, col_mean, 0, CV_REDUCE_AVG);

    cv::Mat centered(_numeric_mat_valid.size(), CV_64F);
    for(int i=0; i < _numeric_mat_valid.rows; i++)
        centered.row(i) = _numeric_mat_valid.row(i) - col_mean;
    cv::Mat cov = (centered.t() * centered) / double(_numeric_mat_valid.rows - 1);

    cv::Mat eigenvalues;
    cv::eigen(cov, eigenvalues);

    double sum = cv::sum( eigenvalues )[0];
    _MFs["Fract1"] = eigenvalues.at<double>(0, 0) / sum;

    return 0;
}

int Data::calClassSD_CV_Skew() {
    double sum_std = 0;
    cv::Mat cv_combined_mat(_class_label.size(), _numeric_mat.cols, CV_64F);
    cv::Mat skew_combined_mat(_class_label.size(), _numeric_mat.cols, CV_64F);
	int row_ind = 0;

    for(set<int>::iterator it=_class_label.begin(); it!= _class_label.end(); it++) {
        int c = *it;
        // get all rows of class c
        int numc = std::count (_class_value.begin(), _class_value.end(), c);
        cv::Mat cmat(numc, _numeric_mat.cols, CV_64F);
        int ind = 0;
        for(int i=0; i < _class_value.size(); i++) {
            if(_class_value[i] == c)
                _numeric_mat.row(i).copyTo(cmat.row(ind++));
        }
        double mean, std, skew, kurtosis;
        for (int i = 0; i < cmat.cols; i++){
            calStats(cmat.col(i), mean, std, skew, kurtosis);
            cv_combined_mat.at<double>(row_ind, i) = std / mean;
            skew_combined_mat.at<double>(row_ind, i) = skew;
            sum_std += std;
        }
        row_ind++;
    }

    _MFs["ClSD.Mean"] = sum_std / (_class_label.size() * _numeric_mat.cols);

    cv::Mat col_mat;
    cv::reduce(cv_combined_mat, col_mat, 0, CV_REDUCE_MAX);
    _MFs["ClCV.Max"] = cv::mean(col_mat)[0];
    _MFs["ClCV.Mean"] = cv::mean(cv_combined_mat)[0];
	
    cv::reduce(skew_combined_mat, col_mat, 0, CV_REDUCE_MIN);
    _MFs["ClSkew.Min"] = cv::mean(col_mat)[0];
    _MFs["ClSkew.Mean"] = cv::mean(skew_combined_mat)[0];

    return 0;
}

double Data::entropyEmpirical(map< vector<int>, int > frequencies, int nsamples) {
	double e = 0;
	for (map< std::vector<int>, int>::const_iterator iter = frequencies.begin(); iter != frequencies.end(); ++iter)
		e -= iter->second * log((double)iter->second);
	return log((double)nsamples) + e / nsamples;
}

double Data::entropy(const cv::Mat d) {
    map< vector<int>, int > freq;
	vector<int> sel;

	int nsamples_ok = 0;

    for (int i = 0; i < d.rows; i++) {
		bool ok = true;
		sel.clear();
        for (int j = 0; j < d.cols; j++) {
            if (isnan(d.at<double>(i, j)))
				ok = false;
			else
                sel.push_back((int)d.at<double>(i, j));
		}
        if (ok) {
			freq[sel]++;
			nsamples_ok++;
		}
	}
	
	return entropyEmpirical(freq, nsamples_ok);
}

double Data::entropy(const vector<int> d) {
    cv::Mat mat(d.size(), 1, CV_64F);
    for(int i=0; i < d.size(); i++)
        mat.at<double>(i, 0) = d[i];
	return entropy(mat);
}

// information theoretic MFs
int Data::calNomalizedEntropy() {
    int N = _discretize_mat.cols;
    cv::Mat entropy_att(N, 1, CV_64F);

	for (int i = 0; i < N; i++) {
		double e = entropy(_discretize_mat.col(i)) / log(double(_n_instances));
        entropy_att.at<double>(i, 0) = e;
	}
	
    double min_val, max_val;
    cv::minMaxLoc(entropy_att, &min_val, &max_val);
    _MFs["entAttN.Min"] = min_val;
    _MFs["entAttN.Max"] = max_val;

	return 0;
}

int Data::calClassEntropy() {
	_MFs["entClassN"] = entropy(_class_value) / log(double(_n_instances));

	return 0;
}

int Data::calJointEntropy() {
   
    int N = _discretize_mat.cols;

    cv::Mat joint_mat(_n_instances, 2, CV_64F);
    _data_mat.col(_numeric_mat.cols).copyTo(joint_mat.col(1));

    cv::Mat joint_entropy_values;
    joint_entropy_values.create(N, 1, CV_64F);

    for (int i = 0; i < N; i++) {
        _discretize_mat.col(i).copyTo(joint_mat.col(0));
        
        for(int r=0; r < joint_mat.rows; r++)
            if(isnan(joint_mat.at<double>(r, 0)))
                joint_mat.at<double>(r, 0) = 100;

		joint_entropy_values.at<double>(i, 0) = entropy(joint_mat);
    }
    
    double mean, std, skew, kurtosis;
    calStats(joint_entropy_values, mean, std, skew, kurtosis);
    _MFs["jEntAC.Mean"] = mean;
    _MFs["jEntAC.SD"] = std;
    _MFs["jEntAC.Skew"] = skew;

	return 0;
}

int Data::calMutualInfo() {
    vector<double> mut_values;
    int N = _discretize_mat.cols;

    cv::Mat joint_mat(_n_instances, 2, CV_64F);
    _data_mat.col(_numeric_mat.cols).copyTo(joint_mat.col(0));

    for (int i = 0; i < N; i++) {
        _discretize_mat.col(i).copyTo(joint_mat.col(1));
		double Hyx = entropy(joint_mat);
		double Hx = entropy(joint_mat.col(0));
		double Hy = entropy(joint_mat.col(1));
		double res = Hx + Hy - Hyx;
		if (res < 0)
			res = 0;
		mut_values.push_back(res);
	}

	double sum = std::accumulate(mut_values.begin(), mut_values.end(), 0.0);
	double mean = sum / N;

	double temp = 0;
	for (int i = 0; i < N; i++)
		temp += (mut_values[i] - mean)*(mut_values[i] - mean);

	double std;
    if (N <= 1) {
		cout << "Warning: number of atrribute <=1, cannot calculate std" << endl;
		std = 0;
	}
	else
		std = sqrt(temp / (N-1));

	_MFs["mutInfoAC.Mean"] = mean;
	_MFs["mutInfoAC.SD"] = std;

	return 0;
}

int Data::calNoiseSignalRatio() {
    int N = _discretize_mat.cols;
	
	double mean_entropy_att = 0;
    for (int i = 0; i < N; i++) {
		mean_entropy_att += entropy(_discretize_mat.col(i));
	}
	mean_entropy_att /= N;

	double mean_mut = 0;
    if (_MFs.find("mutInfoAC.Mean") != _MFs.end()) {
		mean_mut = _MFs["mutInfoAC.Mean"];
    }
    else {
        cv::Mat joint_mat(_n_instances, 2, CV_64F);
        _data_mat.col(_numeric_mat.cols).copyTo(joint_mat.col(0));

        for (int i = 0; i < N; i++) {
            _discretize_mat.col(i).copyTo(joint_mat.col(1));
			double Hyx = entropy(joint_mat);
			double Hx = entropy(joint_mat.col(0));
			double Hy = entropy(joint_mat.col(1));
			double res = Hx + Hy - Hyx;
			if (res < 0)
				res = 0;
			mean_mut += res;
		}
		mean_mut /= N;
	}

	_MFs["NSratio"] = (mean_entropy_att - mean_mut) / mean_mut;

	return 0;
}

// landmarking MFs
vector<int> Data::kfolds(int k) {
    vector<int> folds;
    folds.clear();
    if (k == 1) {
        for(int i=0; i < _n_instances; i++)
            folds.push_back(1);
    }else {
        float i = (float)_n_instances / k;
        if(i < 1) {
            cout << "insufficient records: " << _n_instances << " with k= " << k << endl;
            return folds;
        }

        vector<int> vec_i;
        vec_i.push_back(0);
        for(int ind=1; ind < k; ind++)
            vec_i.push_back((int)myroundf(i*ind));
        vec_i.push_back(_n_instances);

        vector<int> times;
        for(int ind=0; ind < vec_i.size()-1; ind++)
            times.push_back(vec_i[ind+1]-vec_i[ind]);

        for(int ind=0; ind < times.size(); ind++)
        {
            for(int j=0; j < times[ind]; j++)
                folds.push_back(ind);
        }

        //std::srand ( unsigned ( time(0) ) );
        std::srand( 10 );
        std::random_shuffle ( folds.begin(), folds.end() );
    }

    return folds;
}

int Data::calErrorRate(cv::Mat pred, cv::Mat tr, map<int, double> &err) {
    //cout << "true: " << tr << " pred: " << pred << endl;
    for(set<int>::iterator it=_class_label.begin(); it!= _class_label.end(); it++) {
        int c = *it;
        double fp = 0, fn = 0;
        for(int i=0; i < pred.rows; i++) {
            if((int)tr.at<float>(i, 0) != c && (int)pred.at<float>(i, 0) == c)
                fp++;
            if((int)tr.at<float>(i, 0) == c && (int)pred.at<float>(i, 0) != c)
                fn++;
        }
        err[c] += (fp+fn) / pred.rows;
        //cout << "class " << c << " fp: " << fp << " fn: " << fn << " err: " << (fp+fn) / pred.rows << " acc err: " << err[c] << endl;
    }
    return 0;
}

int Data::calLandmarking() {
    int k = 10;

    vector<int> folds = kfolds(k);
    if (folds.size() == 0) {
        _MFs["DN"] = -1;
        if(_MFs_mask["WN"])
            _MFs["WN"] = -1;
        return 1;
    }

    bool cal_wn = _MFs_mask["WN"];

    int N = _numeric_mat.cols;
    int max_level = cal_wn ? N : 1;

    CvDTreeParams params = CvDTreeParams(
        max_level, // max depth
        5,// min sample count
        0, // regression accuracy: N/A here
        true, // compute surrogate split, no missing data
        10, // max number of categories (use sub-optimal algorithm for larger numbers)
        0, // the number of cross-validation folds
        false, // use 1SE rule => smaller tree
        false, // throw away the pruned tree branches
        NULL // the array of priors
    );

    CvDTreeParams params1 = CvDTreeParams(
        1, 5, 0, true, 5, 0, false, false, NULL
    );

    cv::Mat var_type = cv::Mat(N+1, 1, CV_8U );
    for(int i=0; i < N; i++) {
        if(_header[i].type == NUMERICAL)
            var_type.at<unsigned char>(i, 0) = CV_VAR_NUMERICAL;
        else {
            if(_header[i].num_category < 100)
                var_type.at<unsigned char>(i, 0) = CV_VAR_CATEGORICAL;
            else
                var_type.at<unsigned char>(i, 0) = CV_VAR_NUMERICAL;
        }
    }
    var_type.at<unsigned char>(N, 0) = CV_VAR_CATEGORICAL;

    //make sure data doens't contain NA
    cv::Mat data_mat_no_na;   
    if(_has_na && _na_option == NA_KEEP) {
        _data_mat.copyTo(data_mat_no_na);
        for(int r=0; r < _data_mat.rows; r++)
            for(int c=0; c < _data_mat.cols; c++)
                if(isnan(_data_mat.at<double>(r,c)))
                    data_mat_no_na.at<double>(r,c) = 10000;
    } else {
        data_mat_no_na = _data_mat;
    }

    // find best and worst nodes
    CvDTree dtree;
    cv::Mat train_all;
    data_mat_no_na.convertTo(train_all, CV_32F); 
    dtree.train(train_all(cv::Range(0, train_all.rows), cv::Range(0,N)), CV_ROW_SAMPLE,
                 train_all.col(N), cv::Mat(), cv::Mat(), var_type, cv::Mat(), params); 
    cv::Mat var_importance = dtree.getVarImportance();

   //cout << "var importance: " << var_importance << endl;
    cv::Point best_node_idx, worst_node_idx;
    double min_val, max_value;
    cv::minMaxLoc(var_importance, &min_val, &max_value, &worst_node_idx, &best_node_idx);
    //cout << "best node: " << best_node_idx.x << endl;
    //cout << "worst node: " << worst_node_idx.x << endl << endl;

    cv::Mat vt_best = cv::Mat(2, 1, CV_8U );
    cv::Mat vt_worst = cv::Mat(2, 1, CV_8U );
    vt_best.at<unsigned char>(1, 0) = CV_VAR_CATEGORICAL;
    vt_worst.at<unsigned char>(1, 0) = CV_VAR_CATEGORICAL;
    
    if(_header[best_node_idx.x].type == NUMERICAL)
        vt_best.at<unsigned char>(0, 0) = CV_VAR_NUMERICAL;
    else {
        if(_header[best_node_idx.x].num_category < 100)
            vt_best.at<unsigned char>(0, 0) = CV_VAR_CATEGORICAL;
        else
            vt_best.at<unsigned char>(0, 0) = CV_VAR_NUMERICAL;
    }

    if(_header[worst_node_idx.x].type == NUMERICAL)
        vt_worst.at<unsigned char>(0, 0) = CV_VAR_NUMERICAL;
    else {
        if(_header[worst_node_idx.x].num_category < 100)
            vt_worst.at<unsigned char>(0, 0) = CV_VAR_CATEGORICAL;
        else
            vt_worst.at<unsigned char>(0, 0) = CV_VAR_NUMERICAL;
    }

    // maps to store accumulated error rate
    map<int, double> err_dn, err_wn;
    for(set<int>::iterator it=_class_label.begin(); it!= _class_label.end(); it++) {
        int c = *it;
        err_dn[c] = 0;
        err_wn[c] = 0;
    }

    cv::Mat testvalue(1, 1, CV_32F);
    for(int i=0; i < k; i++) {
        //cout << "fold: " << i << endl;
        int n_test = std::count (folds.begin(), folds.end(), i);
        int n_train = folds.size() - n_test;
        cv::Mat test(n_test, _n_attributes, CV_32F);
        cv::Mat train(n_train, _n_attributes, CV_32F);
        int test_ind = 0, train_ind = 0;
        for(int r=0; r < _n_instances; r++) {
            if(folds[r] == i)
                train_all.row(r).copyTo(test.row(test_ind++));
            else
                train_all.row(r).copyTo(train.row(train_ind++));
        }
        //cout << "train rows: " << train.rows << endl;
        //cout << "test rows: " << test.rows << endl;
        //cout << endl << train << endl;
        //cout << test << endl;
        /*
        mexPrintf("train\n");
        for(int rr=0; rr < n_train; rr++) {
            for(int cc=0; cc < _n_attributes; cc++)
                mexPrintf("%0.2f ", train.at<float>(rr,cc));
            mexPrintf("\n");
        }
        mexPrintf("n_train: %d, _n_attributes: %d\n", n_train, _n_attributes);
        mexPrintf("train_ind: %d\n", train_ind);
        mexPrintf("train.rows: %d, train.cols %d\n", train.rows, train.cols);
        */
        //DN & WN
        CvDTree dtree_dn, dtree_wn;
        dtree_dn.train(train.col(best_node_idx.x), CV_ROW_SAMPLE,
                     train.col(N), cv::Mat(), cv::Mat(), vt_best, cv::Mat(), params1);
        if(cal_wn)
            dtree_wn.train(train.col(worst_node_idx.x), CV_ROW_SAMPLE,
                     train.col(N), cv::Mat(), cv::Mat(), vt_worst, cv::Mat(), params1);

        cv::Mat pred_mat_dn(test.rows, 1, CV_32F);
        cv::Mat pred_mat_wn(test.rows, 1, CV_32F);
        for (int testid = 0; testid < test.rows; testid++) {
            testvalue.at<float>(0, 0) = test.at<float>(testid, best_node_idx.x);
            pred_mat_dn.at<float>(testid, 0) = (float)dtree_dn.predict(testvalue)->value;
            if(cal_wn) {
                testvalue.at<float>(0, 0) = test.at<float>(testid, worst_node_idx.x);
                pred_mat_wn.at<float>(testid, 0) = (float)dtree_wn.predict(testvalue)->value;
            }
        }

        calErrorRate(pred_mat_dn, test.col(N), err_dn);
        if(cal_wn)
            calErrorRate(pred_mat_wn, test.col(N), err_wn);
    }
    // calculate mean
    double dn_mean = 0, wn_mean = 0;
    for(set<int>::iterator it=_class_label.begin(); it!= _class_label.end(); it++) {
        int c = *it;
        //cout << "dn_err class " << c << ": " << dn_err[c] << " and weight: " << _class_weight[c] << endl;
        dn_mean += (err_dn[c] / k)*_class_weight[c];
        if(cal_wn)
            wn_mean += (err_wn[c] / k)*_class_weight[c];
    }
    _MFs["DN"] = dn_mean;
    if(cal_wn)
        _MFs["WN"] = wn_mean;

    return 0;
}

// concept characterisation
int Data::calConceptChar() {

    cv::Mat nmat; // _numeric_mat after removing rows having NA and constant columns
    _numeric_mat_valid.copyTo(nmat);
    
    // remove contant columns
    vector<int> non_constant_col_ind;
    for(int c=0; c < nmat.cols; c++) {
        bool constant = true;
        for(int r=0; r < nmat.rows-1; r++)
            if (nmat.at<double>(r, c) != nmat.at<double>(r+1, c)) {
                constant = false;
                break;
            }
        if(!constant)
            non_constant_col_ind.push_back(c);
    }
    if(non_constant_col_ind.size() != nmat.cols) { // has constant columns
        cv::Mat temp;
        temp.create(nmat.rows, non_constant_col_ind.size(), CV_64F);
        int ind = 0;
        for(int i=0; i < non_constant_col_ind.size(); i++) {
            nmat.col(non_constant_col_ind[i]).copyTo(temp.col(ind++));
        }
        nmat = temp;
    } 
  
    // calculate U (upper bound) and L (lower bound) for columns
    cv::Mat col_mean;
    cv::reduce(nmat, col_mean, 0, CV_REDUCE_AVG);

    // ccMFpreproc
    cv::Mat st_dat_mat(nmat.rows, nmat.cols, CV_64F);
    cv::Mat xU, xL;
    cv::reduce(nmat, xU, 0, CV_REDUCE_MAX);
    cv::reduce(nmat, xL, 0, CV_REDUCE_MIN);
    for(int i=0; i < nmat.cols; i++) {
        cv::Mat col = nmat.col(i);
        col = (col - (xL.at<double>(0, i) + xU.at<double>(0, i))/2) / ((xU.at<double>(0, i) - xL.at<double>(0, i))/2);
        col.copyTo(st_dat_mat.col(i));
    }
    
    double min_val, max_val;
  
    cv::Mat U, L;
    cv::reduce(st_dat_mat, U, 0, CV_REDUCE_MAX);
    cv::reduce(st_dat_mat, L, 0, CV_REDUCE_MIN);
    cv::Mat UL = U - L;

    //generate D matrix
    cv::Mat d_mat;

    unsigned int start_t = getTime();

    double* vg_d = NULL;

#ifdef USE_CUDA
    if(_n_instances > 8000) {
        std::vector<double> vg_a;
        if (st_dat_mat.isContinuous()) {
            vg_a.assign((double*)st_dat_mat.datastart, (double*)st_dat_mat.dataend);
        } else {
            for (int i = 0; i < st_dat_mat.rows; ++i) {
                vg_a.insert(vg_a.end(), (double*)st_dat_mat.ptr<uchar>(i), (double*)st_dat_mat.ptr<uchar>(i)+st_dat_mat.cols);
            }
        }
        std::vector<double> vg_ul;
        for(int i = 0; i < UL.cols; i++)
            vg_ul.push_back(UL.at<double>(0, i));
        vg_d = new double[_n_instances*_n_instances];

        int pitch_a = st_dat_mat.cols * sizeof(double);
        int pitch_d = _n_instances * sizeof(double);
        distanceGPU(   &vg_a[0], pitch_a, _n_instances, 
                    &vg_ul[0], st_dat_mat.cols, 
                    vg_d, pitch_d);
        d_mat = cv::Mat(_n_instances, _n_instances, CV_64F, vg_d);
    }
    else {
        d_mat = calDistanceMatrix(st_dat_mat, UL);
    }

#else
    d_mat = calDistanceMatrix(st_dat_mat, UL);
#endif

#ifdef PRINT_DEBUG
    cout << "Time to create dmat: " << getTime() - start_t << endl;
#endif

    //cout << endl << "d_mat: " << endl << d_mat << endl;
    
    int N = nmat.rows;

    //generate delta matrix
    start_t = getTime();
    cv::Mat delta_mat(N, N, CV_64F);
    if(_na_option == NA_KEEP && _has_na) {
        for(int i=0; i < N; i++) {
            int c = _class_value_valid[i];
            for(int j=0; j < N; j++)
                delta_mat.at<double>(i, j) = (c == _class_value_valid[j]) ? 0 : 1; 
        }

    } else {
        for(int i=0; i < N; i++) {
            int c = _class_value[i];
            for(int j=0; j < N; j++)
                delta_mat.at<double>(i, j) = (c == _class_value[j]) ? 0 : 1;
        }
    }
    //cout << endl << "delta_mat: " << endl << delta_mat << endl;
#ifdef PRINT_DEBUG
    cout << "Time to create delta_mat: " << getTime() - start_t << endl;
#endif

    start_t = getTime();
    double alpha = 2;
    //cv::minMaxLoc(d_mat, &min_val, &max_val);
    max_val = -HUGE_VAL;
    for(int i=0; i < N; i++)
        for(int j=0; j < N; j++)
            if(d_mat.at<double>(i, j) > max_val)
                max_val = d_mat.at<double>(i, j);
#ifdef PRINT_DEBUG
    cout << "Time to find max val: " << getTime() - start_t << endl;
#endif    
    cv::Mat D(N, N, CV_64F);
    cv::Mat W(N, N, CV_64F);

    double sqrt_n_instances = sqrt(double(N));

    //D
    start_t = getTime();
#ifdef USE_OpenMP
#pragma omp parallel for
#endif
    for(int i=0; i < N; i++)
        for(int j=0; j < N; j++) {
            D.at<double>(i, j) = d_mat.at<double>(i, j) / max_val * sqrt_n_instances;
        }
#ifdef PRINT_DEBUG
    cout << "Time to calculate D: " << getTime() - start_t << endl;
#endif
    
    //W
    start_t = getTime();
#ifdef USE_OpenMP
#pragma omp parallel for
#endif
    for(int i=0; i < N; i++)
        for(int j=0; j < N; j++) {
            if(i == j)
                W.at<double>(i, j) = 0;
            else {
                W.at<double>(i, j) = alpha*D.at<double>(i, j) / (sqrt_n_instances - D.at<double>(i, j));
                W.at<double>(i, j) = 1 / pow(2, W.at<double>(i, j));
                if(isnan(W.at<double>(i, j)))
                    W.at<double>(i, j) = 0;
            }  
        }
#ifdef PRINT_DEBUG
    cout << "Time to calculate W: " << getTime() - start_t << endl;
#endif
    
    start_t = getTime();
    cv::Mat W_delta; 
    cv::multiply(W, delta_mat, W_delta);
#ifdef PRINT_DEBUG
    cout << "Time to calculate W_delta: " << getTime() - start_t << endl;
#endif
    //cout << endl << "W_delta: " << endl << W_delta << endl;
    start_t = getTime();
    cv::Mat num; cv::reduce(W_delta, num, 1, CV_REDUCE_SUM);
    cv::Mat den; cv::reduce(W, den, 1, CV_REDUCE_SUM);
    cv::Mat cv; cv::divide(num, den, cv);
#ifdef PRINT_DEBUG
    cout << "Time to calculate numm den cv: " << getTime() - start_t << endl;
#endif
    //cout << endl << "num: " << num << endl;
    //cout << "den: " << den << endl;
    //cout << "cv: " << cv << endl;
    //printMat("den: ", den);

    start_t = getTime();
    cv::Mat W_D; cv::multiply(W, D, W_D);
#ifdef PRINT_DEBUG
    cout << "Time to calculate W_D: " << getTime() - start_t << endl;
#endif
    cv::reduce(W_D, num, 1, CV_REDUCE_SUM);
    cv::Mat wd; cv::divide(num, den, wd);
    //cout << "wd: " << wd << endl;

    double mean, std, skew, kurtosis;
    calStats(cv, mean, std, skew, kurtosis);
    _MFs["conceptVariation.Mean"] = mean;
    _MFs["conceptVariation.StdDev"] = std;
    _MFs["conceptVariation.Kurtosis"] = kurtosis;

    calStats(wd, mean, std, skew, kurtosis);
    _MFs["weightedDist.Mean"] = mean;
    _MFs["weightedDist.StdDev"] = std;
    _MFs["weightedDist.Skewness"] = skew;

    if(vg_d)
        delete [] vg_d;

    return 0;
}
