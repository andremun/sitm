#if _WIN32
#include <conio.h>
#endif

#include "Data.h"
#include <iostream>

using namespace std;

void usage() {
    cout << endl;
    cout << "Usage: ./metafeature [-h] -i arff_input_file [-n nan_option] [-o result_output_file]" << endl;
    cout << "   -i: /path/to/inputfile.arff" << endl;
    cout << "   -o: /path/to/outputfile.txt" << endl;
    cout << endl;
}

int main(int argc, const char* argv[]) {
    if (argc <= 2)
	{
        usage();
		return 1;
	}

    string infilename, outfilename;
    bool print_to_file = false;
    int na_option = NA_KEEP;

    int i = 1;
    while (i < argc)
    {
        if (strcmp(argv[i],"-h")==0) {
            usage();
            return 1;
        }
        else if (strcmp(argv[i],"-i")==0) {
            infilename = argv[++i];
            i++;
        }
        else if (strcmp(argv[i],"-o")==0) {
            outfilename = argv[++i];
            print_to_file = true;
            i++;
        }
        else if (strcmp(argv[i],"-n")==0) {
            na_option = atoi(argv[++i]);
            i++;
        }
    }

	Data* data = new Data(na_option);
    const bool feature_mask[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0};
    const bool feature_mask_f[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    int ret = data->loadDataFromArffFile((const char*)infilename.c_str(), feature_mask_f);
    //int ret = data->loadDataFromArffFile((const char*)infilename.c_str(), feature_mask);
	if (ret) 
	{
		cout << "Fail to load data from file " << infilename << endl;
		return 1;
	}

    ofstream ofs;
    if(print_to_file) {
       ofs.open(outfilename.c_str(), ofstream::out);
       if(ofs == 0)
           cout << "WARNING: cannot open file " << outfilename << " to write" << endl;
    }
    else
        ofs.open("", ofstream::out); // NULL
	
    // print data info
    data->print(true, ofs);

    unsigned int t = data->getTime();
    data->calMFs();
    cout << "calMFs time: " << data->getTime() - t << endl << endl;
    data->printMFs(ofs);
    
    if(ofs)
        ofs.close();
	cout << endl << "Done!" << endl;

#if _WIN32
	_getch();
#endif

	return 0;
}
