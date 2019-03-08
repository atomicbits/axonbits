//
// Created by Peter Rigole on 2019-03-08.
//
#include <string.h>

#include "Managed.cuh"

#ifndef INC_01_GETTING_STARTED_STRING_H
#define INC_01_GETTING_STARTED_STRING_H


// String Class for Managed Memory
class String : public Managed {
public:
    String();

    // Constructor for C-string initializer
    String(const char*);

    // Copy constructor
    String(const String&);

    ~String();

    // Assignment operator
    String &operator=(const char*);

    // Element access (from host or device)
    __host__ __device__ char& operator[](int);

    // C-string access
    __host__ __device__ const char* c_str() const;

private:
    int length;
    char* data;
    void _realloc(int);

};


#endif //INC_01_GETTING_STARTED_STRING_H
