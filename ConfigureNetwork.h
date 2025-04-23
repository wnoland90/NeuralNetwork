#ifndef CONFIGURENETWORK_H
#define CONFIGURENETWORK_H

#include <string>
#include <vector>

struct ConfigureNetwork {
    std::vector<std::vector<std::vector<double> > > manualWeights = {};
    std::vector<std::vector<double> > manualBiases = {};
    double l1 = 0.0;
    double l2 = 0.0;
    float alpha = 0.001;
    float dropOutRate = 0.0;
    bool training = true;
    float beta1 = 0.0;

    ConfigureNetwork() {};

    ConfigureNetwork(
        std::vector<std::vector<std::vector<double> > > manualWeights_,
        std::vector<std::vector<double> > manualBiases_,
        double l1_,
        double l2_,
        float alpha_,
        float dropOutRate_,
        bool training_,
        float beta1_
        ) : manualWeights(manualWeights_), manualBiases(manualBiases_), l1(l1_), l2(l2_), alpha(alpha_), dropOutRate(dropOutRate_), training(training_), beta1(beta1_) {}

};

#endif
