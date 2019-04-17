//
// Created by Peter Rigole on 2019-04-17.
//

#include "NeuronProperties.cuh"

NeuronProperties::NeuronProperties() : long_time_lambda(0.5),
                                       medium_time_lambda(0.5) {}

// Copy constructor
NeuronProperties::NeuronProperties(const NeuronProperties &neuronProperties) :
        long_time_lambda(neuronProperties.long_time_lambda),
        medium_time_lambda(neuronProperties.medium_time_lambda) {}

// Destructor
NeuronProperties::~NeuronProperties() {}

__host__ __device__
float NeuronProperties::getLongTimeLambda() const { return long_time_lambda; }

__host__ __device__
float NeuronProperties::getMediumTimeLambda() const { return medium_time_lambda; }
