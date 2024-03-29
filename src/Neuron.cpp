//
// Author: marvin on 11/18/19.
//

#include "Neuron.h"


/*
eta - overall net learning rate
 0.0 - slow learner
 0.2 - medium learner
 1.0 - reckless learner

alpha - momentum
 0.0 - no momentum
 0.5 - moderate momentum
*/
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;
int Neuron::numNeurons = 0;

void Neuron::updateInputWeights(vector<Neuron> &prevLayer) {
    //The weights to be updated are in the Connection container
    //in the neurons in the preceding layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDetalWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                *neuron.getOutputVal()
                *m_gradient
                //Also add momentum = a fraction of the previous delta weight
                + alpha
                + oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDetalWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDetalWeight;
    }
}

double Neuron::sumDOW(const vector<Neuron> &nextLayer) const {
    double sum = 0.0;
    //Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() -1; ++n) {
        sum += m_outputWeights[n].weight*nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const vector<Neuron> &nextLayer) {
    double down = sumDOW(nextLayer);
    m_gradient = down*activationFunctionDerivative(m_outputVal);
}


void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta*activationFunctionDerivative(m_outputVal);
}

double Neuron::activationFunction(double x) {
    //than - output range [-1.0..1.0]
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    //than derivative
    return 1.0 - pow(x, 2);
}

void Neuron::feedForward(const vector<Neuron> &prevLayer) {
    double sum = 0.0;
    //Sum the previous layer's output (which are our inputs)
    //Include the bias node from the previous layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal()*
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    //activation function
    m_outputVal = activationFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}



