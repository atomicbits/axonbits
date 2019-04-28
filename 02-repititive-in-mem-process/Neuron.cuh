//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H

#include "Managed.cuh"
#include "NeuronProperties.cuh"
#include "Synapse.cuh"
#include "util/Array.cu"
#include "Phase.cu"
#include "CycleParity.cu"


class NeuronProperties; // forward declaration to cope with cyclic dependency
class Synapse; // forward declaration to cope with cyclic dependency

class Neuron : public Managed {
public:

    /**
     * Creates a memory managed neuron with the given id and neuron properties. It reserves the space to store
     * max_nb_incoming_synapses number of synapse pointers in a managed array.
     *
     * @param neuronId
     * @param neuronProperties
     * @param max_nb_incoming_synapses
     */
    Neuron(unsigned long int neuronId,
            NeuronProperties* neuronProperties,
            unsigned int max_nb_incoming_excitatory_synapses,
            unsigned int max_nb_incoming_inhibitory_synapses);

    // Destructor
    ~Neuron();

    // Get the id
    __host__ __device__
    unsigned long int getId() const;

    __host__ __device__
    const NeuronProperties* getProperties() const;

    __host__ __device__
    float getActivity(CycleParity parity) const;

    /**
     * Update the activity of this neuron, which sets its new activity.
     * This does not override the current activity while in this cycle! So, getActivity(currentCycleParity) will
     * return the same value before and after updateActivity(...) !!! The updated activity will be used as the
     * current activity in the next cycle.
     *
     * The effect of this way of handling the activity updates using cycle parity is that the single thread that does
     * the update of the activity during a cycle will not collide with the threads that request the current (old) value
     * of the neuron's activity to calculate the average input activity in all the neurons that receive from this
     * neuron. So it effectively isolates thread update activity, so avoiding all kinds of synchronization issues.
     */
    __host__ __device__
    void updateActivity(float activity_update, CycleParity parity);

    /**
     * Sets the activity into a neuron as an external input activity.
     * Should only be called from the host as an input signal!
     * It sets the activity for both cycle parities.
     */
    __host__
    void setExternalActivity(float activity_update);

    /**
     * Get the short time average activity based on the length of te fourth quarter cycles.
     * This value only makes sense after the end of the fourth quarter cycles.
     */
    __host__ __device__
    float getShortTimeAverageActivity(const int fourthQuarterLength) const;

    __device__
    void resetShortTimeSumActivity();

    __device__
    void incrementShortTimeSumActivity(const float activity);

    /**
     * Get the medium time average activity based on the length of the first three quarter cycles.
     * This value only makes sense after the end of the first three quarter cycles.
     */
    __host__ __device__
    float getMediumTimeAverageActivity(const int threeQuarterLength) const;

    __device__
    void resetMediumTimeSumActivity();

    __device__
    void incrementMediumTimeSumActivity(const float activity);

    __host__ __device__
    float getLongTimeAverageActivity() const;

    __device__
    void incrementLongTimeAverageActivity(const float activity, const float alpha);

    __host__ __device__
    Array<Synapse>* getIncomingExcitatorySynapses() const;

    __host__ __device__
    void addIncomingExcitatorySynapse(Synapse* synapse);

    __host__ __device__
    Array<Synapse>* getIncomingInhibitorySynapses() const;

    __host__ __device__
    void addIncomingInhibitorySynapse(Synapse* synapse);

private:
    unsigned long int id;
    const NeuronProperties* properties;
    /**
     * y(t-1) is the current activity, which was calculated in the previous cycle.
     * y(t-1) will be used in the calculation of g_e(t) = 1/n sum_i(x_i(t-1)*w_i), where x_i(t-1) is the previously
     * calculated activity of the source neurons of each incoming synapse.
     *
     * y(t) is the new activity, which is calculated in the current cycle based on y(t-1) and g_e(t).
     *
     */
    float activity_even_parity; // either y(t) or y(t-1) depending on the current cycle parity
    float activity_odd_parity;  // either y(t) or y(t-1) depending on the current cycle parity
    // y_s averaged during the last 25ms (during the last quarter of a trial)
    float short_time_sum_activity;
    // y_m averaged during the first 75ms of each trial
    float medium_time_sum_activity;
    // y_l averaged using running exponential average
    float long_time_avg_activity;
    // incoming excitatory synapses
    Array<Synapse>* incoming_excitatory_synapses;
    // incoming inhibitory synapses
    Array<Synapse>* incoming_inhibitory_synapses;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
