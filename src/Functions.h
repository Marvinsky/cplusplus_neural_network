

#ifndef WORKSPACE_FUNCTIONS_H
#define WORKSPACE_FUNCTIONS_H

#include "Tipos.h"

void showVectorVals(string label, vector<double> &v) {
    cout<<label<<" ";
    for(unsigned i = 0; i < v.size(); ++i) {
        cout<<v[i]<<" ";
    }
    cout<<endl;
}

#endif //WORKSPACE_FUNCTIONS_H
