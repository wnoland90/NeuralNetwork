#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <cmath>

#include "NeuralLayer.h"
#include "SetupLayers.h"
#include "ConfigureNetwork.h"

using namespace std;
// using json = nlohmann::json;

class NeuralNetwork {

    private:

        bool CheckManualWeights(vector<SetupLayers>, vector<vector<vector<double> > > mw);

    public:
        // NeuralNetwork();
        void MakeNeuralNetwork(vector<SetupLayers> layerData_, ConfigureNetwork cn);

        int numInputNeurons;

        int numLayers;

        double alpha;

        // float epsilon;

        float lambda1;

        bool l1;

        float lambda2;

        bool l2;

        float dropOutRate;

        double errorCost;
        
        vector<NeuralLayer> networkLayers;

        vector<vector<vector<double> > > weightsList;

        vector<vector<vector<double> > > SetupWeights(vector<SetupLayers> layerData, vector<vector<vector<double> > > manualWeights_);

        vector<double> ForwardPass(vector<double> input);

        void BackwardPass(vector<double> correction);

        void UpdateTheWeights();

        void TurnTrainingOff();

        // void SaveNetwork(string fileName="");

};

#endif
