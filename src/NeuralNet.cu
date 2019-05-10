//
// Created by Peter Rigole on 2019-03-13.
//
#include <stdio.h> // Needed for printf.
#include "NeuralNet.cuh"

// = = = = = = = = = = = = = = = = = =
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
float avgInputActivity(NeuralNet *neuralNet, Array<Synapse>* synapses, const CycleParity parity) {
    int number = 0;
    float sum = 0.0;
    for(Array<Synapse>::iterator i = synapses->begin(); i != synapses->end(); ++i) {
        Synapse* synapse = &*i;
        float weight = synapse->getWeight();
        float activity = neuralNet->getNeuron(synapse->getSource())->getActivity(parity);
        sum += activity * weight;
        ++number;
    }
    if(number == 0) return 0.0;
    else return sum / (float)number;
}

/**
 * XCAL dWt function
 */
__device__
float XCALdWt(const float xy, const float theta_p) {
    float theta_d = 0.1;
    if (xy > theta_p * theta_d) {
        return xy - theta_p;
    } else {
        return - xy * (1 - theta_d) / theta_d;
    }
}

__device__
float contrastEnhancement(const float x) {
    float gain = 2.5;   // (gamma)
    float offset = 1.0; // (theta)

    return 1 / (1 + powf((offset * (1 - x)/x), gain));
}

/**
 * Update the incoming synapse weights based on the short-term, medium-term and long-term average activity of the
 * source and target neuron (the given neuron).
 */
__device__
void updateIncomingSynapsesWeights(NeuralNet* neuralNet, Neuron* neuron, const CycleParity parity) {

    float y_l_lrn_min = 0.005;
    float y_l_lrn_max = 0.05;
    float y_l_max = 1.5;
    float y_l_min = 0.2;
    float y_l = neuron->getLongTimeAverageActivity();
    float y_m = neuron->getMediumTimeAverageActivity(75); // ToDo: refactor out the magic number
    float y_s = neuron->getShortTimeAverageActivity(25);  // ToDo: refactor out the magic number

    // Effective learning reate for y_l
    float y_l_lrn = y_l_lrn_min + (y_l - y_l_min) * ((y_l_lrn_max - y_l_lrn_min)/y_l_max - y_l_min);
    float y_m_lrn = 1.0;

    float learningRate = 0.1;

    Array<Synapse>* synapses_exc = neuron->getIncomingExcitatorySynapses();
    Array<Synapse>* synapses_inh = neuron->getIncomingInhibitorySynapses();

    for(Array<Synapse>::iterator i = synapses_exc->begin(); i != synapses_exc->end(); ++i) {
        Synapse* synapse = &*i;

        float x_s = neuralNet->getNeuron(synapse->getSource())->getShortTimeAverageActivity(25);  // ToDo: refactor out the magic number
        float x_m = neuralNet->getNeuron(synapse->getSource())->getMediumTimeAverageActivity(75); // ToDo: refactor out the magic number

        float xy_s = x_s * y_s;
        float xy_m = x_m * y_m;

        float dwt = learningRate * (y_m_lrn * XCALdWt(xy_s, xy_m) + y_l_lrn * XCALdWt(xy_s, y_l));

        // Weight bounding
        float weight = synapse->getWeight();
        if (dwt > 0) {
            weight += (1 - weight) * dwt;
        } else {
            weight += weight * dwt;
        }
        // Contrast enhancement
        weight = contrastEnhancement(weight);

        // Update the weight in the synapse
        synapse->updateWeight(weight);
    }


    // Not sure how we should handle inhibitory synaptic plasticity...
    // references:
    //  - https://www.cell.com/neuron/fulltext/S0896-6273(12)00771-4
    //  - https://www.frontiersin.org/articles/10.3389/fncir.2013.00119/full
    //  - https://www.frontiersin.org/articles/10.3389/fncel.2014.00093/full
    //
    // Should we go for an overall inhibitory function such as a simplified FFFB (feedforward & feedback) function?
    // Or some K-winner-takes-all approach?
    // Or should we add some inhibitory neurons with fixed weights to play the same role? This has the advantage that
    // it is completely distributed and is easily to integrate in our GPU-based solution!
    //    for(Array<Synapse>::iterator i = synapses_inh->begin(); i != synapses_inh->end(); ++i) {
    //        Synapse* synapse = *i;
    //
    //    }



}

__device__
void updateNeuronActivity(NeuralNet* neuralNet,
                          Neuron* neuron,
                          const Phase phase,
                          const bool beginOfPhase,
                          const bool endOfPhase,
                          const CycleParity parity,
                          const int outcomePhDuration) {

    if (beginOfPhase) {
        if (phase == ExpectationPhase) {
            neuron->resetMediumTimeSumActivity();
        } else if (phase == OutcomePhase) {
            neuron->resetShortTimeSumActivity();
        }
    }

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
    float g_i_rel = avgInputActivity(neuralNet, neuron->getIncomingInhibitorySynapses(), parity);
    float g_i = g_bar_i * g_i_rel;

    float g_e_theta = (g_i * (E_i - theta) + g_l * (E_l - theta)) / (theta - E_e);

    // g_e_rel = g_e(t) = 1/n * sum_i(x_i(t) * w_i)
    float g_e_rel = avgInputActivity(neuralNet, neuron->getIncomingExcitatorySynapses(), parity);
    float g_e = g_bar_e * g_e_rel;

    float y_current = neuron->getActivity(parity);
    float y_star = 1 / (1 + 1 / (lambda * positivePart(g_e - g_e_theta))); // ToDo: add gaussian noise to this function!
    float y = y_current + dt_vm * (y_star - y_current); // new activity for the neuron

    neuron->updateActivity(y, parity);

    if (phase == ExpectationPhase) {
        neuron->incrementMediumTimeSumActivity(y);
    } else if (phase == OutcomePhase) {
        neuron->incrementShortTimeSumActivity(y);
        if (endOfPhase) {
            // Update the long time average activity based on the most recent short time average activity.
            float alpha = 0.02;
            neuron->incrementLongTimeAverageActivity(neuron->getShortTimeAverageActivity(outcomePhDuration), alpha);
        }
    }
}

__global__
void cycleParallelized(NeuralNet* neuralNet,
                       Array<Neuron>* neurons,
                       const Phase phase,
                       const bool beginOfPhase,
                       const bool endOfPhase,
                       const CycleParity parity,
                       const int outcomePhDuration) {
    // Using the popular grid-stride loop
    // see: https://devblogs.nvidia.com/even-easier-introduction-cuda/
    // see: https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned int nbOfNeurons = neurons->getSize();
    for (int i = index; i < nbOfNeurons; i += stride) {
        Neuron* neuron = &((*neurons)[i]);
        updateNeuronActivity(neuralNet, neuron, phase, beginOfPhase, endOfPhase, parity, outcomePhDuration);
    }
}

__global__
void updateWeightsParallelized(NeuralNet* neuralNet, Array<Neuron>* neurons, const CycleParity parity) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned int nbOfNeurons = neurons->getSize();
    for (int i = index; i < nbOfNeurons; i += stride) {
        Neuron* neuron = &((*neurons)[i]);
        updateIncomingSynapsesWeights(neuralNet, neuron, parity);
    }
}


// End of globally scoped functions
// = = = = = = = = = = = = = = = = = =


NeuralNet::NeuralNet() : max_nb_of_threads(256), max_nb_of_blocks(4096) {
    nb_of_threads = max_nb_of_threads;
    nb_of_blocks = max_nb_of_blocks;
}

NeuralNet::NeuralNet(unsigned int maxNeurons_init) : max_nb_of_threads(256), max_nb_of_blocks(4096) {
    nb_of_threads = max_nb_of_threads;
    nb_of_blocks = max_nb_of_blocks;
    neurons = new Array<Neuron>(maxNeurons_init);
}

NeuralNet::~NeuralNet() {
    delete neurons;
}

__host__
void NeuralNet::addNeuron(Neuron &neuron, unsigned int index) {
    neurons->set(neuron, index);
}

__host__ __device__
Neuron* NeuralNet::getNeuron(unsigned long int neuronIndex) {
    return &((*neurons)[neuronIndex]);
}

__host__
void NeuralNet::init() {
    cudaDeviceSynchronize();
}

__host__
void NeuralNet::trial() {

    for (int i = 0; i < (expectationPhaseDuration + outcomePhaseDuration); i++) {
        bool beginOfPhase = i == 0 || i == expectationPhaseDuration;
        bool endOfPhase = i == expectationPhaseDuration - 1 || i == expectationPhaseDuration + outcomePhaseDuration - 1;
        Phase currentPhase;
        if (i < expectationPhaseDuration) {
            currentPhase = ExpectationPhase;
        } else {
            currentPhase = OutcomePhase;
        }
        if (i % 100 == 0) process10HzInput();
        if ((i + 1) % 100 == 0) process10HzOutput();

        cycle(currentPhase, beginOfPhase, endOfPhase, getParity(i));
    }

    // The cycle containing the last updated activities is one cycle after the last one, which was an OddCycle.
    updateWeights(EvenCycle); // Could this be done in parallel with reading the neural net output signals and writing the new input signals?
}

__host__
void NeuralNet::process10HzInput() {
    inputProcessor10Hz->processInput();
}

__host__
void NeuralNet::process10HzOutput() {
    outputProcessor10Hz->processOutput();
}

__host__
void NeuralNet::register10HzInputProcessor(InputProcessor* inputProcessor_update) {
    inputProcessor10Hz = inputProcessor_update;
}

__host__
void NeuralNet::register10HzOutputProcessor(OutputProcessor* outputProcessor_update) {
    outputProcessor10Hz = outputProcessor_update;
}

__host__
void NeuralNet::getActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId) const {
    // Assume the last activity has an odd cycle parity.
    int activityIndex = 0;
    for (Array<Neuron>::iterator i = neurons->index(fromNeuronId); i != neurons->index(toNeuronId + 1); ++i) {
        Neuron* neuron = &*i;
        activity[activityIndex] = neuron->getActivity(OddCycle);
        ++activityIndex;
    }
    return;
}

__host__
void NeuralNet::updateActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId) {
    int activityIndex = 0;
    for (Array<Neuron>::iterator i = neurons->index(fromNeuronId); i != neurons->index(toNeuronId + 1); ++i) {
        Neuron* neuron = &*i;
        neuron->setExternalActivity(activity[activityIndex]);
        ++activityIndex;
    }
    return;
}

__host__
void NeuralNet::cycle(const Phase phase, const bool beginOfPhase, const bool endOfPhase, const CycleParity parity) {
    // Divide the neurons over a whole bunch of threads and get them to work...
    // ToDo: see which number of blocks and block size work best...
    // Test with block size of 1024 (i.e. number of threads per block).
    // Mind that cycleParallelized(...) must be called as a function, not a method! CUDA constraint...

    cycleParallelized<<<nb_of_blocks,nb_of_threads>>>(this, neurons, phase, beginOfPhase, endOfPhase, parity, outcomePhaseDuration);
    cudaDeviceSynchronize();

    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Device failure during activation update cycles, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }
}

__host__
void NeuralNet::updateWeights(const CycleParity parity) {
    // Divide the neurons over a whole bunch of threads and get them to work...

    updateWeightsParallelized<<<nb_of_blocks,nb_of_threads>>>(this, neurons, parity);
    cudaDeviceSynchronize();

    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Device failure during weight updating, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }

    return;
}

__host__
CycleParity NeuralNet::getParity(int i) {
    if(i % 2 == 0) return EvenCycle;
    else return OddCycle;
}

__host__
void NeuralNet::initThreadBlocks() {
    unsigned int nb_of_neurons = neurons->getSize();
    if (nb_of_neurons < max_nb_of_threads) {
        nb_of_threads =  nb_of_neurons;
        nb_of_blocks = 1;
    } else if (nb_of_neurons / max_nb_of_threads < max_nb_of_blocks) {
        if (nb_of_neurons % max_nb_of_threads == 0) {
            nb_of_blocks = nb_of_neurons / max_nb_of_threads;
        } else {
            nb_of_blocks = (nb_of_neurons / max_nb_of_threads) + 1;
        }
    }
}
