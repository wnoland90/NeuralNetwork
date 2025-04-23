#ifndef NEURALLAYER_H
#define NEURALLAYER_H

#include <string>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <cmath>
#include <random>


using namespace std;

class NeuralLayer {

    private:

        vector<double> ApplyValuesForward(vector<vector<double> > weights, vector<double> value);
        vector<double> ApplyValuesBackward(vector<vector<double> > weights, vector<double> values);
        vector<double> Activation();
        vector<double> ActivationDerivative();

    public:

        unsigned int numNeurons;                    // Number of Neurons in this layer.

        string activationFunction;                  // The activation function of this layer, options are: "relu", "linear" and "sigmoid", more to come later.

        vector<double> preActivatedNeurons;         // This layer's values before applying the activation function.

        vector<double> postActivatedNeurons;        // This layer's values after applying the activation function.

        string layerName;                           // This layer's name.

        vector<double> biases;                      // Each neuron's bias of this layer.

        vector<double> errorSignals;                // When checking the error rate of the back propagation this vector will keep track of the error signals.

        double alpha;                               // The learning rate of the network, this will be inherited from the network.

        float dropOutRate;                          // The rate of which each neuron might get dropped from the network during training (to prevent the network to focus on one feature/trait).

        vector<bool> notDropped;                    // Neurons that were dropped during the training. The vector will be of all 1's and 0's.

        bool isTraining;                            // Used to check if this network is still training or not.

        void BuildThisLayer(int numNeurons_, string activationFunction_, string layerName_, float alpha_, float dropOutRate_, vector<double> manualBiases_={});
        void ForwardPass(vector<vector<double> > weights, vector<double> values);
        void BackPropagation(vector<vector<double> > weights, vector<double> values);
        vector<double> SetupBiases(vector<double> manualBiases);
        void UpdateTheBiases();
        vector<bool> DroppedNeurons();

};

#endif
