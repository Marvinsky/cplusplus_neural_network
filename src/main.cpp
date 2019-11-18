
#include "functions.h"
#include "TrainingData.h"
#include "Network.h"

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

    return 0;
}