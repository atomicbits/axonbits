//
// Created by Peter Rigole on 2019-03-08.
//
#include <stddef.h>

#ifndef AXONBITS_MANAGED_H
#define AXONBITS_MANAGED_H


// Managed Base Class -- inherit from this to automatically
// allocate objects in Unified Memory
class Managed {
public:

    __host__
    void *operator new(size_t);

    __host__
    void operator delete(void *);
};


#endif //AXONBITS_MANAGED_H
