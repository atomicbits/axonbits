//
// Created by Peter Rigole on 2019-04-26.
//

#ifndef AXONBITS_CYCLEPARITY_H
#define AXONBITS_CYCLEPARITY_H

/**
 * All threads working on a cycle have the same cycle parity. This parity is used to identify the activity variable
 * in the neuron that is to be updated (the next activity) versus the one that must be used as the neuron's current
 * activity.
 */
enum CycleParity { EvenCycle, OddCycle };

#endif //AXONBITS_CYCLEPARITY_H
