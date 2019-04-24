//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H

#include "Managed.cuh"
#include "Neuron.cuh"
#include "util/Array.cu"

enum Phase { ExpectationPhase, OutcomePhase };

class NeuralNet : public Managed {

public:

    /**
     * Constructor
     *
     * @param maxNeurons_init max number of neurons
     *
     * unsigned int: 0 to 4,294,967,295 (see: https://www.geeksforgeeks.org/c-data-types/)
     *
     */
    NeuralNet(unsigned int maxNeurons_init);

    // Destructor
    ~NeuralNet();

    __host__
    void trial();

    __host__
    void getActivity(float[], unsigned int fromNeuronId, unsigned int toNeuronId) const;

    __host__
    void updateActivity(float[], unsigned int fromNeuronId, unsigned int toNeuronId);


private:

    __host__ __device__
    void cycle(const Phase phase);

    __host__ __device__
    void updateWeights();

    Array<Neuron>* neurons;

};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H
