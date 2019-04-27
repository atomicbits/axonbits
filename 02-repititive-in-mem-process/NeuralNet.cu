//
// Created by Peter Rigole on 2019-03-13.
//
#include <stdio.h> // Needed for printf.
#include "NeuralNet.cuh"

// = = =
// Some functions defined on the global scope (because calling OO methods as parallel CUDA function doesn't work).
// See: https://stackoverflow.com/questions/40558908/cuda-illegal-combination-of-memory-qualifiers
// PS: This remark has nothing to do with the __global__ qualifier below, that's CUDA stuff. Defining functions on the
// global scope means they don't sit in an object as a method.

__device__
float positivePart(float x) {
    if(x >= 0) return x;
    else return 0;
}

/**
 * g_e(t) = 1/n * sum_i(x_i(t) * w_i)
 */
__device__
float avgyInputActivity(Array<Synapse>* synapses, const CycleParity parity) {
    int number = 0;
    float sum = 0.0;
    for(Array<Synapse>::iterator i = synapses->begin(); i != synapses->end(); ++i) {
        Synapse* synapse = *i;
        float weight = synapse->getWeight();
        float activity = synapse->getSource()->getActivity(parity);
        sum += activity * weight;
        ++number;
    }
    if(number == 0) return 0.0;
    else return sum / (float)number;
}

__device__
void updateNeuronActivity(Neuron* neuron, const Phase phase, const CycleParity parity) {

    // Normalized neuron parameters
    // See: https://grey.colorado.edu/CompCogNeuro/index.php/CCNBook/Main
    // ToDo: move them to the NeuronProperties
    float E_e = 1.0; // excitatory driving potential, bio val: 0mV
    float E_i = 0.25; // inhibitory driving potential, bio val: -75mV
    float E_l = 0.3; // leak driving potential, bio val: -70mV
    float g_bar_e = 1.0; // maximum excitatory conductance, bio val: 100nS
    float g_bar_i = 1.0; // maximum inhibitory conductance, bio val: 100nS
    float g_l = 0.1; // constant leak conductance, bio val: 10nS
    float theta = 0.5; // equilibrium Vm at threshold (Vm is the membrane potential), bio val: -50mV
    float lambda = 1.0; // gain factor
    float dt_vm = 0.355; // the rate constant (determines how fast the membrane potential changes) = 1/C, bio val: C = 281pF

    // g_i_rel should be calculated as defined in FFFB inhibition function, or a better and more local way?
    // can we use local inhibition neurons to replace FFFB in a simple way?
    float g_i_rel = avgyInputActivity(neuron->getIncomingInhibitorySynapses(), parity);
    float g_i = g_bar_i * g_i_rel;

    float g_e_theta = (g_i * (E_i - theta) + g_l * (E_l - theta)) / (theta - E_e);

    float g_e_rel = avgyInputActivity(neuron->getIncomingExcitatorySynapses(), parity); // g_e_rel = g_e(t) = 1/n * sum_i(x_i(t) * w_i)
    float g_e = g_bar_e * g_e_rel;

    float y_current = neuron->getActivity(parity);
    float y_star = 1 / (1 + 1 / (lambda * positivePart(g_e - g_e_theta))); // ToDo: add gaussian noise to this function!
    float y = y_current + dt_vm * (y_star - y_current); // new activity for the neuron

    neuron->updateActivity(y, parity);
}

__global__
void cycleParallelized(Array<Neuron>* neurons, const Phase phase, const CycleParity parity) {
    // Using the popular grid-stride loop
    // see: https://devblogs.nvidia.com/even-easier-introduction-cuda/
    // see: https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned int nbOfNeurons = neurons->getSize();
    for (int i = index; i < nbOfNeurons; i += stride) {
        Neuron* neuron = (*neurons)[i];
        updateNeuronActivity(neuron, phase, parity);
    }
}


// End of globally scoped functions
// = = =


NeuralNet::NeuralNet(unsigned int maxNeurons_init)  {
    neurons = new Array<Neuron>(maxNeurons_init);
}

NeuralNet::~NeuralNet() {
    delete neurons;
}

__host__
void NeuralNet::trial() {
    for (int i = 0; i < 75; i++) {
        cycle(ExpectationPhase, getParity(i));
    }
    for (int i = 75; i < 100; i++) {
        cycle(OutcomePhase, getParity(i));
    }
    updateWeights(); // Could this be done in parallel with reading the neural net output signals and writing the new input signals?
}

__host__
void NeuralNet::getActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId) const {
    // Assume the last activity has an odd cycle parity.
    return;
}

__host__
void NeuralNet::updateActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId) {
    return;
}

__host__
void NeuralNet::cycle(const Phase phase, const CycleParity parity) {
    // Divide the neurons over a whole bunch of threads and get them to work...
    // ToDo: see which number of blocks and block size work best...
    // Test with block size of 1024 (i.e. number of threads per block).
    // Mind that cycleParallelized(...) must be called as a function, not a method! CUDA constraint...
    cycleParallelized<<<4096,256>>>(neurons, phase, parity);
    cudaDeviceSynchronize();

    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Device failure, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }
}

__host__
void NeuralNet::updateWeights() {
    // Divide the neurons over a whole bunch of threads and get them to work...

    return;
}

__host__
CycleParity NeuralNet::getParity(int i) {
    if(i % 2 == 0) return EvenCycle;
    else return OddCycle;
}
