//
// Author: marvin on 11/18/19.
//

#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"

class Net {
private:
    vector<vector<Neuron>> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
public:
    Net(const vector<unsigned> &topology);
    ~Net(){}
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const {return m_recentAverageError;}
};

#endif //NETWORK_H
