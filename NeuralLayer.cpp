#include "NeuralLayer.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>


vector<double> NeuralLayer::SetupBiases(vector<double> manualBiases) {
    //      FUNC SetupBiases() creates the bias parameters for each neuron in this layer. Takes in no variables.
    //      Strictly used to initialize the neurons' biases.
    //          vector<double> manualBiases_: default is empty, but if its full this could be used for a specific function to setup an already made neural layer.
    
    if (manualBiases.empty() != true) {
        if (manualBiases.size() == numNeurons) {
            return manualBiases;
        }
    }
    vector<double> biasList(numNeurons, 0.0);
        srand(time(0));
        int x;
        for (size_t i = 0; i < numNeurons; i++) {
            x = rand();
            biasList[i] = (x % 100000) / 1000000.0;

            if (x % 9 == 0 or x % 7 == 0) {
                biasList[i] = biasList[i] * -1;
            }
            srand(x + (time(0) * i));
        }
        return biasList;
        
}


vector<double> NeuralLayer::ApplyValuesForward(vector<vector<double> > weights, vector<double> values) {
    //      FUNC ApplyValuesForward(weights, values) creates a vector of pre-activated values for this layer for a forward pass.
    //          vector<vector<double> > weights: the weight matrix of N x M (N=previous layer count (call it 'col'), M=this layer count (call it 'row')) parameters between the layers.
    //          vector<double> values: the values of the layer going into this layer.

    vector<double> outputList(numNeurons, 0.0);
    for (size_t row = 0; row < numNeurons; row++) {
        for (size_t col = 0; col < values.size(); col++) {
            outputList[row] = outputList[row] + (weights[col][row] * values[col]);
        }
        outputList[row] = outputList[row] + biases[row];
    }
    return outputList;
}


vector<double> NeuralLayer::ApplyValuesBackward(vector<vector<double> > weights, vector<double> values) {
    //      FUNC ApplyValuesBackward(weights, values) creates a vector of pre-activated values for this layer for a backward pass.
    //          vector<vector<double> > weights: the weight matrix of N x M (N=this layer count (call it 'row'), M=next layer count (call it 'col')) parameters between the layers.
    //          vector<double> values: the values of the layer going into this layer.

    vector<double> outputList(numNeurons, 0.0);
    for (size_t col = 0; col < values.size(); col++) {
        for (size_t row = 0; row < numNeurons; row++) {
            outputList[row] = outputList[row] + (weights[row][col] * values[col]);
        }
    }
    return outputList;
}


vector<double> NeuralLayer::Activation() {
    //      FUNC Activation() creates a vector of post-activated values for this layer.

    vector<double> outList(numNeurons, 0.0);
    // double softmaxTotal = 0.0;
    // if (activationFunction == "softmax") {
    //     for (size_t z = 0; z < numNeurons; z++) {
    //         softmaxTotal = softmaxTotal + exp(preActivatedNeurons[z]);
    //     }
    // }

    for (size_t i = 0; i < numNeurons; i++) {
        if (notDropped[i] == true or isTraining == false) {
            if ((activationFunction == "relu" and preActivatedNeurons[i] > 0) or activationFunction == "linear") {
                outList[i] = preActivatedNeurons[i];
            } else if (activationFunction == "sigmoid") {
                outList[i] = 1 / (1 + exp(-1 * preActivatedNeurons[i]));
            // } else if (activationFunction == "softmax") {
            //     outList[i] = exp(preActivatedNeurons[i]) / softmaxTotal;
            }
        }
    }
    return outList;
}


vector<double> NeuralLayer::ActivationDerivative() {
    //      FUNC ActDerivative() creates a vector of the derivatives of this layer
    //      ===============NEED TO ADD SOFTMAX DERIVATIVE TO THIS FUNCTION=================

    vector<double> output(numNeurons, 0.0);
    for (size_t i = 0; i < numNeurons; i++) {
        if ((activationFunction == "relu" and preActivatedNeurons[i] > 0) or (activationFunction == "linear")) {
            output[i] = 1.0;
        } else if (activationFunction == "sigmoid") {
            output[i] = (postActivatedNeurons[i] * (1 - postActivatedNeurons[i]));
        }
    }
    return output;
}


void NeuralLayer::BuildThisLayer(int numNeurons_, string activationFunction_, string layerName_, float alpha_, float dropOutRate_, vector<double> manualBiases_){
    //      FUNC BuildThisLayer(numNeurons_, activationFunction_, layerName_) sole purpose is to initialize this layer and the neurons.
    //          int numNeurons_: the number of neurons in this layer.
    //          string activationFunction_: the activation function that will be used in this layer.
    //          string layerName_: name of the layer.
    //          vector<double> manualBiases_: default is empty, but if its full this could be used for a specific function to setup an already made neural layer.

    numNeurons = numNeurons_;
    activationFunction = activationFunction_;
    layerName = layerName_;
    alpha = alpha_;
    dropOutRate = dropOutRate_;
    notDropped = vector<bool>(numNeurons, true);
    isTraining = true;
    preActivatedNeurons = vector<double>(numNeurons, 0.0);
    postActivatedNeurons = vector<double>(numNeurons, 0.0);
    errorSignals = vector<double>(numNeurons, 0.0);
    biases = SetupBiases(manualBiases_);
}


void NeuralLayer::ForwardPass(vector<vector<double> > weights, vector<double> values) {
    //      FUNC ForwardPass(weights, values) used for applying the values from the previous layer to this layer.
    //          vector<vector<double> > weights: the weight matrix of N x M (N=previous layer count, M=this layer count) parameters between the layers.
    //          vector<double> values: the values of the layer going into this layer.
    
    if (isTraining == true) {
        notDropped = DroppedNeurons();
    }
    preActivatedNeurons = ApplyValuesForward(weights, values);
    postActivatedNeurons = Activation();
}


void NeuralLayer::BackPropagation(vector<vector<double> > weights, vector<double> values) {
    //      FUNC BackPropagation(weights, values) used for applying the values from the next layer's errors to this layer.
    //          vector<vector<double> > weights: the weight matrix of N x M (N=previous (the next layer) layer count, M=this layer count) parameters between the layers.
    //          vector<double> values: the values of the layer going into this layer.

    vector<double> derivatives = ActivationDerivative();
    vector<double> sumErrorValues = ApplyValuesBackward(weights, values);
    for (size_t i = 0; i < numNeurons; i++) {
        if (notDropped[i] == true) {
            errorSignals[i] = derivatives[i] * sumErrorValues[i];
        } else {
            errorSignals[i] = 0.0;
        }
        
    }
    UpdateTheBiases();
}


void NeuralLayer::UpdateTheBiases() {
    //      FUNC UpdateTheBiases() used for updating the biases of this layer after backpropagation.

    for (size_t i = 0; i < numNeurons; i++) {
        if (notDropped[i] == true) {
            biases[i] = biases[i] - (alpha * errorSignals[i]);
        }
    }
}


vector<bool> NeuralLayer::DroppedNeurons() {
    //      FUNC DroppedNeurons() makes a new vector of true and false that will determine if the neuron will be dropped depending on dropOutRate.

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);

    vector<bool> tmpDropped(numNeurons, true);
    for (int s = 0; s < numNeurons; s++) {
        if (dist(gen) < dropOutRate) {
            tmpDropped[s] = false;
        }
    }
    return tmpDropped;
}
