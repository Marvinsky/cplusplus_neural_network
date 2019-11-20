//
// Author: marvin on 11/18/19.
//

#ifndef NEURON_H
#define NEURON_H

#include "Connection.h"

class Neuron {
private:
    static double eta; //[0.0..1.0] overall net training rate
    static double alpha; //[0.0..n] multiplier of last weight change (momentum)
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void) {rand()/double(RAND_MAX);}
    double sumDOW(const vector<Neuron> &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    ~Neuron(){}
    void setOutputVal(double val) {m_outputVal = val;}
    double getOutputVal(void) const {return m_outputVal;}
    void feedForward(const vector<Neuron> &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const vector<Neuron> &nextLayer);
    void updateInputWeights(vector<Neuron> &prevLayer);
};

#endif //NEURON_H
