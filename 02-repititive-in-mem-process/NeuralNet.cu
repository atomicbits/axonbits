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
void updateNeuronActivity(Neuron* neuron, const Phase phase, const CycleParity parity) {

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
