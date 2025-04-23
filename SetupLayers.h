#ifndef SETUPLAYERS_H
#define SETUPLAYERS_H

#include <string>

struct SetupLayers {
    int previous;
    int current;
    std::string activationFunction;
    std::string name;
    
    SetupLayers(
        int previous_,
        int current_,
        std::string act,
        std::string name_
        ) : previous(previous_), current(current_), activationFunction(act), name(name_) {}
};

#endif
