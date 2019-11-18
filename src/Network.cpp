//
// Author: marvin on 11/18/19.
//

#include "Network.h"

void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size() -1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

Net::Net(const vector<unsigned int> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(vector<Neuron>());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        //we have made a new layer, now fill it ith neurons, and
        //add a BIAS neuron to the layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout<<"Made a Neuron"<<endl;
        }
        //force the bias node's output value to 1.0, it's the last neuron created above
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const vector<double> &inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1);
    //Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); i++) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        vector<Neuron> &prevLayer = m_layers[layerNum-1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals) {
    //calculate overall net error (RMS of output neurons errors)
    vector<Neuron> &outputLayer = m_layers.back();
    m_error = 0.0;
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta*delta;
    }
    m_error /= outputLayer.size() - 1;//get average error squared
    m_error = sqrt(m_error);//RMS

    //Implement a recent average measurement
    m_recentAverageError =
            (m_recentAverageError*m_recentAverageSmoothingFactor+m_error)
            /(m_recentAverageSmoothingFactor + 1.0);
    //calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    //calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size() -2; layerNum > 0; --layerNum) {
        vector<Neuron> &hiddenLayer = m_layers[layerNum];
        vector<Neuron> &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    //for all layers from outputs to first hidden layer,
    //update connections weights
    for (unsigned layerNum = m_layers.size() -1; layerNum > 0; --layerNum) {
        vector<Neuron> &layer = m_layers[layerNum];
        vector<Neuron> &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

