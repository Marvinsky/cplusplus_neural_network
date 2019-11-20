//
// Author: marvin on 11/20/19.
//

#ifndef GENERATION_H
#define GENERATION_H

#include "Tipos.h"

class Generation {
private:
    ofstream file;
public:
    Generation(string filename) {
        file.open(filename);
        if (file.is_open()) {
            //random training set for XOR -- two inputs and one output
            file<<"topology: 2 4 1"<<endl;
            for(int i = 2000; i >= 0; --i) {
                int n1 = (int)(2.0*rand()/double(RAND_MAX));
                int n2 = (int)(2.0*rand()/double(RAND_MAX));
                int t = n1^n2;//should be 0 or 1
                file<<"in: "<<n1<<".0 "<<n2<<".0 "<<endl;
                file<<"out: "<<t<<".0"<<endl;
            }
        }
        file.close();
    }
    ~Generation(){}
};

#endif //GENERATION_H
