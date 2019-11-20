
#include "Functions.h"
#include "TrainingData.h"
#include "Network.h"
#include "Generation.h"
#include <thread>
#include <memory>
#include <future>

int main() {
    string fileNamePath = "../data/trainingData.txt";

    Generation generation(fileNamePath);
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    shared_ptr<TrainingData> trainData(new TrainingData);
    thread t = thread(&TrainingData::init, trainData, fileNamePath);

    //wait for thread to finish
    t.join();

    //e.g., {3, 2, 1}
    vector<unsigned> topology;
    trainData->getTopology(topology);
    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    int trainingError = 0, accuracy;
    while (!trainData->isEof()) {
        ++trainingPass;
        cout<<endl<<"Pass "<<trainingPass;

        //Get new input data and feed it forward:
        if (trainData->getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs: ", inputVals);
        myNet.feedForward(inputVals);

        //collect the net's actual results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs: ", resultVals);

        //Train the net what the outputs should have been:
        trainData->getTargetOutputs(targetVals);
        showVectorVals("Targets: ", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        //report how well the training is working, averaged over recent
        int recentAverageError = myNet.getRecentAverageError();
        cout<<"Net recent average error: "
        <<recentAverageError<<endl;
        if (recentAverageError == 0) {
            accuracy++;
        } else {
            trainingError++;
        }
    }

    cout<<endl<<endl;
    cout<<"Percentage of Accuracy = "
    <<(double(accuracy)/trainingPass)*100<<"%"<<endl;

    cout<<"Percentage of Training Error = "
    <<((double)trainingError/trainingPass)*100<<"%"<<endl;

    cout<<endl<<"Done"<<endl;

    return 0;
}