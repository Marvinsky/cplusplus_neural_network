
#ifndef WORKSPACE_TRAININGDATA_H
#define WORKSPACE_TRAININGDATA_H

#include "Tipos.h"

class TrainingData {
private:
    ifstream m_trainingDataFile;
public:
    TrainingData(){}
    ~TrainingData(){}
    void init(const string filename);
    bool isEof(ifstream trainingFileName) {return m_trainingDataFile.eof();}
    bool isEof(void) {return m_trainingDataFile.eof();}
    void getTopology(vector<unsigned> &topology);

    //returns the number of input values read from the file
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);
};

void TrainingData::getTopology(vector<unsigned> &topology) {
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss>>label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while(!ss.eof()) {
        unsigned n;
        ss>>n;
        topology.push_back(n);
    }
    return;
}

/*
TrainingData::TrainingData(const string filename) {
    m_trainingDataFile.open(filename.c_str());
}*/

void TrainingData::init(const string filename) {
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals) {
    inputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>>label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss>>oneValue) {
            inputVals.push_back(oneValue);
        }
    }
    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals) {
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>>label;
    if(label.compare("out:") == 0) {
        double oneValue;
        while (ss>>oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }
    return targetOutputVals.size();
}


#endif //WORKSPACE_TRAININGDATA_H
