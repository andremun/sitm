#ifndef __DATA_H__
#define __DATA_H__

#include <vector>
#include <map>
#include <string>
#include <set>
using std::vector;
using std::map;
using std::string;
using std::set;
#include <fstream>

#include <opencv2/opencv.hpp>

#define NUM_FEATURES    29

#define NUMERICAL       0
#define NOMINAL         1
#define CLASS           2

#define NA_KEEP        0
#define NA_REMOVE      1
#define NA_ESTIMATE    2

typedef struct Attribute_t {
    int type;
    map<string, int> str_int;   // for nomial attributes and class
    int num_category;           // for nomial attributes and class
} Attribute;

class Data {

private:
    vector<Attribute>	_header;                        // including class col
    vector<int>			_class_value;                   // class values of samples
    set<int>            _class_label;
    map<int, double>    _class_weight;
    cv::Mat             _data_mat;                      // all-numeric + class matrix (nominal attributes are factorized to numeric)
    cv::Mat             _numeric_mat;                   // all-numeric matrix (data without class column)
    cv::Mat             _discretize_mat;                // calculated based on _all_numeric_mat
    int					_n_instances, _n_attributes;    // _n_attributes includes class col

    bool                _has_na;                        // input data contains NaN?
    int                 _na_option;                     // { NAN_KEEP, NAN_REMOVE, NAN_ESTIMATE }
    bool*               _na_row_mask;                   // 0: row doesn't contain any NaN
    bool*               _na_col_mask;                   // 1: col doesn't contain any NaN
    vector<int>         _class_value_valid;             // class values of rows having no NA
    cv::Mat             _numeric_mat_valid;             // _numeric_mat with valid rows only
   
    static const string _feature_names[NUM_FEATURES];
    map<string, double> _MFs;
    map<string, bool>   _MFs_mask;                      // true: enabled; false: disabled

protected:
    int initData();     // to be called after loading data
    void removeNA();
    double estimateColValue(const vector<double>& valid, int col);
    void estimateNA();

public:
	Data(int nan_option);
	~Data();

    // helpers
    unsigned int getTime();
    int loadDataFromArffFile(const char* filename, const bool* mask);
    int loadDataFromMatlab(const double* data, const int rows, const int cols, const int* col_type, const bool* mask);
    void printMat(const char* name, const cv::Mat mat);
    void printMat(const char* name, const cv::Mat mat, std::ofstream& ofs);
    void print(bool print_matrix, std::ofstream& ofs);
    double getMFbyName(string mf_name);
    void printMFs(std::ofstream& ofs);
    cv::Mat discretizeEquaFreq(const cv::Mat data);
    cv::Mat discretizeEquaFreq(const cv::Mat data, const int nbins);
    int calStats(const cv::Mat col_mat, double &mean, double &std, double &skew, double &kurtosis);
    cv::Mat calDistanceMatrix(const cv::Mat mat, const cv::Mat UL);
    int getNumEnabledFeatures();
    bool isFeatureEnabled(int i);
    double getMFValueByID(int i);
    string getMFStringByID(int i);

    // == calculate MFs ==
	int calMFs();

    // simple MFs
    int calNumInstances();			//nInst (excluding class attribute)
    int calNumAttributes();			//nAtt
    int calNumClasses();			//nClasses
    int calNominalProportion();		//pNom
    int calClassProbability();		//classProbSD
    int calNaProportion();			//pNAinst (proportion of instances with at least one NA)
    int calFract1();				//Fract1
    int calClassSD_CV_Skew();		//ClSD.Mean (SD by class) 
									//ClCV.Max, ClCV.Mean (Coefficient of Variation by Class)
									//ClSkew.Min, ClSkew.Mean (Skewness by Class)

	// calculate discreteized matrix using Equa Freq (ref: discretize function in R infotheo package)
	double entropyEmpirical(map< vector<int>, int > frequencies, int nsamples);
    double entropy(const cv::Mat d);
	double entropy(const vector<int> d);
	
	// information theoretic MFs
	int calNomalizedEntropy();		//entAttN.Min, entAttN.Max
	int calClassEntropy();			//entClassN
	int calJointEntropy();			//jEntAC.Mean, jEntAC.SD, jEntAC.Skew
	int calMutualInfo();			//mutInfoAC.Mean, mutInfoAC.SD
	int calNoiseSignalRatio();		//NSratio

    // landmarking MFs
    vector<int> kfolds(int k=10);
    int calErrorRate(cv::Mat pred, cv::Mat tr, map<int, double> &err);
    int calLandmarking();           //DN, WN

    // concept characterisation
    int calConceptChar();           //conceptVariation.Mean, conceptVariation.StdDev
                                    //conceptVariation.Kurtosis, weightedDist.Mean
                                    //weightedDist.StdDev, weightedDist.Skewness
};

#endif
