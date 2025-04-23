#include "NeuralNetwork.h"
#include "SetupLayers.h"
#include "ConfigureNetwork.h"

#include <time.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
// #include <nlohmann/json.hpp>

// using json = nlohmann::json;


// struct SetupLayers {
//     int previous;
//     int current;
//     std::string activationFunction;
//     std::string name;
// };

// struct ConfigureNetwork {
//     std::vector<std::vector<std::vector<double> > > manualWeights = {};
//     std::vector<<std::vector<double> > manualBiases = {};
//     double l1 = 0.0;
//     double l2 = 0.0;
//     float alpha = 0.001;
//     float dropOutRate = 0.0;
// };



void NeuralNetwork::MakeNeuralNetwork(vector<SetupLayers> layerData_, ConfigureNetwork cn) {
    //      FUNC MakeNeuralNetwork(layerData_, cn) initializes the network itself, creating the weights and layers in this network.
    //          vector<SetupLayers> layerData_: this vector has structs and each struct gives the detail of each layer.
    //          ConfigureNetwork cn: a struct with all the data from alpha, manualWeights, manualBiases, l1, l2 (more to come).

    numInputNeurons = layerData_[0].previous;
    numLayers = layerData_.size();
    alpha = cn.alpha;

    if (cn.l1 == 0.0) {
        l1 = false;
    } else {
        l1 = true;
    }

    if (cn.l2 == 0.0) {
        l2 = false;
    } else {
        l2 = true;
    }

    lambda1 = cn.l1;
    lambda2 = cn.l2;
    dropOutRate = cn.dropOutRate;

    weightsList = SetupWeights(layerData_, cn.manualWeights);

    networkLayers = vector<NeuralLayer>(numLayers);
    for (size_t section = 0; section < numLayers; section++) {
        networkLayers[section] = NeuralLayer();
        if (cn.manualBiases.empty() == false) {
            networkLayers[section].BuildThisLayer(layerData_[section].current, layerData_[section].activationFunction, layerData_[section].name, alpha, cn.dropOutRate, cn.manualBiases[section]);
        } else {
            networkLayers[section].BuildThisLayer(layerData_[section].current, layerData_[section].activationFunction, layerData_[section].name, alpha, cn.dropOutRate);
        }
    }

    if (cn.training == false) {
        TurnTrainingOff();
    }
}


bool NeuralNetwork::CheckManualWeights(vector<SetupLayers> ld, vector<vector<vector<double> > > mw) {
    //      FUNC CheckManualWeights(ld, mw) used for checking if the manual weights will work with this network.
    //          vector<SetupLayers> ld: short for LayerData provides the data for how the network should be built.
    //          vector<vector<vector<double> > > mw: short for ManualWeights, used for checking the manual weights if any.

    if (mw.empty() == true) {
        return false;
    }
    if (mw.size() != numLayers) {
        return false;
    }
    for (size_t i = 0; i < numLayers; i++) {
        if ((mw[i].size() != ld[i].previous) or (mw[i][0].size() != ld[i].current)) {
            return false;
        }
    }
    
    return true;
}


vector<vector<vector<double> > > NeuralNetwork::SetupWeights(vector<SetupLayers> layerData, vector<vector<vector<double> > > manualWeights_) {
    //      FUNC SetupWeights(num, data) creates all the weights at each layer.
    //          int num: the number of layers in this network.
    //          vector<SetupLayers> data: a struct that has the info of each layer with the number of neurons from the previous layer and the current layer.
    //          vector<vector<vector<double> > > manualWeights_: a 3d vector default is empty, used for testing and repeating neural networks.
    
    if (CheckManualWeights(layerData, manualWeights_) == true) {
            return manualWeights_;
    }

    vector<vector<vector<double> > > outputWeightsList(numLayers);
    int x;
    srand(time(0));
    for (size_t l = 0; l < numLayers; l++) {
        vector<vector<double> > tmpLayer1(layerData[l].previous);
        for (size_t i = 0; i < layerData[l].previous; i++) {
            vector<double> tmpLayer2(layerData[l].current, 0.0);
            for (size_t j = 0; j < layerData[l].current; j++) {
                x = rand();
                if ((x % 7 == 0 or x % 9 == 0 or x % 11 == 0)) {
                    tmpLayer2[j] = ((x % 100000) / -100000.0);
                } else {
                    tmpLayer2[j] = ((x % 100000) / 100000.0);
                }
                srand(time(0) + x);
            }
            tmpLayer1[i] = tmpLayer2;
        }
        outputWeightsList[l] = tmpLayer1;
    }
    return outputWeightsList;
}


vector<double> NeuralNetwork::ForwardPass(vector<double> input) {
    //      FUNC ForwardPass(input) the forward propagation of the network, goes through each layer and applies the weights. Outputs the final layer's postActivationNeurons.
    //          vector<double> input: the input values of the network.

    for (size_t i = 0; i < numLayers; i++) {
        if (i == 0) {
            networkLayers[i].ForwardPass(weightsList[i], input);
        } else {
            networkLayers[i].ForwardPass(weightsList[i], networkLayers[i - 1].postActivatedNeurons);
        }
    }
    
    return networkLayers[numLayers - 1].postActivatedNeurons;
}


void NeuralNetwork::BackwardPass(vector<double> correction) {
    //      FUNC BackwardPass(correction) runs the backward propagation for each layer.
    //          vector<double> correction: this the true value of the training set.

    for (size_t i = numLayers - 1; i > 0; i--) {
        if ((networkLayers[i].layerName == "Output") or (networkLayers[i].layerName == "output")) {
            errorCost = 0.0;
            for (size_t j = 0; j < networkLayers[i].numNeurons; j++) {
                errorCost = errorCost + (correction[j] - networkLayers[i].postActivatedNeurons[j]);
            }

            errorCost = (errorCost * errorCost) / 2.0;

            if (l1 == true or l2 == true) {

                //  L1 Regularization in the cost function (also known as the 'Lasso Formula')
                //      gets the sum of all the absolute values of the weights in the network
                //      and then gets the product with the lambda1 value and adds it to the
                //      errorCost. What does this regularization do? It helps with making certain
                //      features in the input more relevant, the math makes some weights go to
                //      exact 0.

                //  L2 Regularization in the cost function (also known as the 'Ridge Formula' or
                //      'Weight Decay Formula') gets the sum of all the squared weights in the
                //      neural network and then gets the product of it with the lambda2. After
                //      halving that product add it to the errorCost. What does this regularization
                //      do? It helps with preventing some weights to go down to 0, the opposite of
                //      L1, this can help with overfitting.

                //  If both L1 and L2 are activated then this is called 'Elastic Net Regularization'
                //      and it takes the benefits of both regularizations.


                double l1_regularization_sum = 0.0;
                double l2_regularization_sum = 0.0;
                for (size_t layer = 0; layer < numLayers; layer++) {
                    for (size_t row = 0; row < weightsList[layer].size(); row++) {
                        for (size_t col = 0; col < weightsList[layer][row].size(); col++) {
                            
                            if (l1 == true) {
                                l1_regularization_sum = l1_regularization_sum + abs(weightsList[layer][row][col]);
                            }

                            if (l2 == true) {
                                l2_regularization_sum = l2_regularization_sum + pow(weightsList[layer][row][col], 2);
                            }
                        }
                    }
                }

                //  Add up all the regularizations to the errorCost even if its just 0.
                errorCost = errorCost + (lambda1 * l1_regularization_sum) + ((lambda2 * l2_regularization_sum) / 2);
            }

            //  Takes in the true value of this training set and get the difference between the
            //      predicted value and the true value.
            vector<double> values(correction.size());
            for (size_t c = 0; c < correction.size(); c++) {
                values[c] = networkLayers[i].postActivatedNeurons[c] - correction[c];
            }

            networkLayers[i - 1].BackPropagation(weightsList[i], values);

        } else {
            
            networkLayers[i - 1].BackPropagation(weightsList[i], networkLayers[i].errorSignals);
        }
    }

    UpdateTheWeights();
}


void NeuralNetwork::UpdateTheWeights() {
    //      FUNC UpdateTheWeights() updates the weights of this section using the alpha (learning rate) and the errors of the previous layer.

    for (size_t l = 0; l < networkLayers.size(); l++){
        for (size_t i = 0; i < weightsList[l].size(); i++) {
            for (size_t j = 0; j < weightsList[l][i].size(); j++) {

                // Checks if any neuron was dropped (to and from).
                bool sourceActive = (l > 0) ? networkLayers[l - 1].notDropped[i] == 1 : true;
                if ((sourceActive == true) and (networkLayers[l].notDropped[j] == true)) {

                    float regularizationSpace = 0.0;

                    if (l1 == true) {

                        //  If this network uses L1 Regularization.

                        int sign = 0;
                        if (weightsList[l][i][j] > 0) {
                            sign = 1;
                        } else if (weightsList[l][i][j] < 0) {
                            sign = -1;
                        }
                        regularizationSpace = regularizationSpace + (sign * lambda1);

                    }


                    if (l2 == true) {

                        //  If this network uses L2 regularization.

                        regularizationSpace = regularizationSpace + (weightsList[l][i][j] * lambda2);
                    
                    }

                    weightsList[l][i][j] = weightsList[l][i][j] - (alpha * (networkLayers[l].errorSignals[j] + regularizationSpace));
                }
            }
        }
    }
}


void NeuralNetwork::TurnTrainingOff() {
    //      FUNC TurnTrainingOff() iterates over every layer and turns off training so more drop outs.

    for (int i = 0; i < numLayers; i++) {
        networkLayers[i].isTraining = false;
    }
}


// void NeuralNetwork::SaveNetwork(string fileName) {
//     //      FUNC SaveNetwork(fileName) saves the weights and biases of the network to a file that we can call back to later.
//     //          string fileName: is optional if the file name is given it will be saved under that name.

//     // if (fileName != "") {
//     //     ofstream file(fileName);
//     // } else {
//     //     auto now = chrono::system_clock::now();
//     //     time_t currentTime = chrono::system_clock::to_time_t(now);

//     //     std::tm* localTime = localtime(&currentTime);

//     //     stringstream ss;

//     //     ss << "saved_" << std::put_time(localTime, "%Y") << std::put_time(localTime, "%m") << std::put_time(localTime, "%d") << ".json";
        
//     //     ofstream file(ss.str());
//     // }

//     nlohmann::json nnData;
//     nnData["data"];

//     for (int layer = 0; layer < numLayers; layer++) {
//         string l = to_string(layer);
//         nnData["data"]["layer"][l]["weights"] = weightsList[layer];
//         nnData["data"]["layer"][l]["biases"] = networkLayers[layer].biases;
//     }

//     nlohmann::json config;
    
//     config["Configuration"];
//     config["Configuration"]["L1"] = lambda1;
//     config["Configuration"]["L2"] = lambda2;
//     config["Configuration"]["alpha"] = alpha;
//     config["Configuration"]["dropout"] = dropOutRate;

//     nnData["data"] = config;

//     std::cout << nnData << std::endl;

// }
