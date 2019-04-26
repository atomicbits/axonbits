//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H

#include "Managed.cuh"
#include "Neuron.cuh"
#include "util/Array.cu"
#include "Phase.cu"
#include "CycleParity.cu"

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

    /**
     * Get the activity of the neurons from id 'fromNeuronId' to id 'toNeuronId'.
     *
     * @param activity the array where the requested activity is stored.
     * @param fromNeuronId the from neuron id
     * @param toNeuronId the to neuron id
     */
    __host__
    void getActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId) const;

    /**
     * Update the activity of all neurons from id 'fromNeuronId' to id 'toNeuronId'. It updates both the
     * even and the odd cycle parity activity value (because it is meant to be used as external input).
     *
     * @param activity the array containing the activity to update in the successive neurons
     * @param fromNeuronId the from neuron id
     * @param toNeuronId the to neuron id
     */
    __host__
    void updateActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId);

    __host__
    void cycle(const Phase phase, const CycleParity parity);


private:

//    __global__
//    void cycleParallelized(const Phase phase, const CycleParity parity);

//    __device__
//    void updateNeuronActivity(Neuron* neuron, const Phase phase, const CycleParity parity);

    __host__
    void updateWeights();

    __host__
    CycleParity getParity(int i);

    /**
     * The neuron array.
     * There is a one-to-one map between the id of each neuron and their index in the array.
     */
    Array<Neuron>* neurons;

};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H
