//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef AXONBITS_NEURALNET_H
#define AXONBITS_NEURALNET_H

#include "Managed.cuh"
#include "Neuron.cuh"
#include "Synapse.cuh"
#include "util/Array.cu"
#include "Phase.cu"
#include "CycleParity.cu"
#include "InputProcessor.cuh"
#include "OutputProcessor.cuh"

class InputProcessor; // forward declaration to cope with cyclic dependency
class OutputProcessor; // forward declaration to cope with cyclic dependency

class NeuralNet : public Managed {

public:

    NeuralNet();

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

    /**
     * Adds a neuron to this neural net.
     *
     * Important!!!
     * The index of this neuron in the 'neurons' array will be the same as the id of this neuron!!!
     * ALWAYS make sure to fill up the 'neurons' array entirely without gaps up to a max neuron id. This max neuron id
     * must be smaller than or equal to the maximum number of neurons allowed in this neural net. This maximum allowed
     * number of neurons is set during construction using 'maxNeurons_init' variable in the constructor.
     * If the largest neuron id (or the largest used index in the 'neurons' array) is not preceded with neurons with a
     * continuous id, from 0 up to this largest neuron id, then the neural net will fail with a memory access error.
     *
     */
    __host__
    void addNeuron(Neuron &neuron, unsigned int index);

    __host__ __device__
    Neuron* getNeuron(unsigned long int neuronIndex);

    __host__
    void init();

    __host__
    void trial();

    __host__
    void register10HzInputProcessor(InputProcessor* inputProcessor_update);

    __host__
    void register10HzOutputProcessor(OutputProcessor* outputProcessor_update);

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
    void initThreadBlocks();

    const int expectationPhaseDuration = 75;
    const int outcomePhaseDuration = 25;

private:

    __host__
    void cycle(const Phase phase, const bool beginOfPhase, const bool endOfPhase, const CycleParity parity);

    __host__
    void updateWeights(const CycleParity parity);

    __host__
    CycleParity getParity(int i);

    __host__
    void process10HzInput();

    __host__
    void process10HzOutput();

    // Input and output processors are not managed now, this should not be necessary.
    InputProcessor* inputProcessor10Hz;
    OutputProcessor* outputProcessor10Hz;

    /**
     * The neuron array.
     * There is a one-to-one map between the id of each neuron and their index in the array.
     */
    Array<Neuron>* neurons;

    int nb_of_threads; // number of threads per block
    int nb_of_blocks;  // number of blocks

    const int max_nb_of_threads; // max number of threads per block
    const int max_nb_of_blocks;  // max number of blocks

};


#endif //AXONBITS_NEURALNET_H
