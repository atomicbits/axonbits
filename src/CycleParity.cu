//
// Created by Peter Rigole on 2019-04-26.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_CYCLEPARITY_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_CYCLEPARITY_H

/**
 * All threads working on a cycle have the same cycle parity. This parity is used to identify the activity variable
 * in the neuron that is to be updated (the next activity) versus the one that must be used as the neuron's current
 * activity.
 */
enum CycleParity { EvenCycle, OddCycle };

#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_CYCLEPARITY_H
