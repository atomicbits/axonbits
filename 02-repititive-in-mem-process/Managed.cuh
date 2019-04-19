//
// Created by Peter Rigole on 2019-03-08.
//
#include <stddef.h>

#ifndef INC_01_GETTING_STARTED_MANAGED_H
#define INC_01_GETTING_STARTED_MANAGED_H


// Managed Base Class -- inherit from this to automatically
// allocate objects in Unified Memory
class Managed {
public:

    __host__ __device__
    void *operator new(size_t);

    __host__ __device__
    void operator delete(void *);
};


#endif //INC_01_GETTING_STARTED_MANAGED_H
