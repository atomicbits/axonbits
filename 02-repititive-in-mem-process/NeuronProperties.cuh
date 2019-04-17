//
// Created by Peter Rigole on 2019-04-17.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURONPROPERTIES_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURONPROPERTIES_H

#include "Managed.cuh"

class NeuronProperties : public Managed {
public:
    // Default Constructor
    NeuronProperties();

    // Copy constructor
    NeuronProperties(const NeuronProperties&);

    // Destructor
    ~NeuronProperties();

    __host__ __device__
    float getLongTimeLambda() const;

    __host__ __device__
    float getMediumTimeLambda() const;


private:
    // long_time_lambda and medium_time_lambda balance between long-term statistical (Self-Organizing, Hebbian)
    // learning and Error-Driven learning. Both are between 0 and 1 and their sum is 1.
    float long_time_lambda;
    float medium_time_lambda;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURONPROPERTIES_H
