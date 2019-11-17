
#include "functions.h"
#include "TrainingData.h"

using namespace std;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

//***************************Neuron************************
class Neuron {
private:
    static double eta; //[0.0..1.0] overall net training rate
    static double alpha; //[0.0..n] multiplier of last weight change (momentum)
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void) {rand()/double(RAND_MAX);}
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) {m_outputVal = val;}
    double getOutputVal(void) const {return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
};

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

void Neuron::updateInputWeights(Layer &prevLayer) {
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

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    //Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() -1; ++n) {
        sum += m_outputWeights[n].weight*nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double down = sumDOW(nextLayer);
    m_gradient = down*Neuron::activationFunctionDerivative(m_outputVal);
}


void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta*Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::activationFunction(double x) {
    //than - output range [-1.0..1.0]
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    //than derivative
    return 1.0 - pow(x, 2);
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;
    //Sum the previous layer's output (which are our inputs)
    //Include the bias node from the previous layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal()*
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    //activation function
    m_outputVal = Neuron::activationFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

//***************************Net***************************
class Net {
private:
    vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const {return m_recentAverageError;}
};

void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size() -1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

Net::Net(const vector<unsigned int> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
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
        Layer &prevLayer = m_layers[layerNum-1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals) {
    //calculate overall net error (RMS of output neurons errors)
    Layer &outputLayer = m_layers.back();
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
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    //for all layers from outputs to first hidden layer,
    //update connections weights
    for (unsigned layerNum = m_layers.size() -1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}



int main() {

    TrainingData trainData("../data/trainingData.txt");
    //e.g., {3, 2, 1}
    vector<unsigned> topology;
    trainData.getTopology(topology);
    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout<<endl<<"Pass "<<trainingPass;

        //Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs: ", inputVals);
        myNet.feedForward(inputVals);

        //collect the net's actual results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs: ", resultVals);

        //Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets: ", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        //report how well the training is working, averaged over recent
        cout<<"Net recent average error: "
        <<myNet.getRecentAverageError()<<endl;
    }
    cout<<endl<<"Done"<<endl;

    /*
    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultVals;
    myNet.getResults(resultVals);
    */

    return 0;
}