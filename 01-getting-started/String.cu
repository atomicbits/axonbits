//
// Created by Peter Rigole on 2019-03-08.
//
#include "String.cuh"

String::String() : length(0), data(0) {}

// Constructor for C-string initializer
String::String(const char *s) : length(0), data(0) {
    _realloc(strlen(s));
    strcpy(data, s);
}

// Copy constructor
String::String(const String &s) : length(0), data(0) {
    _realloc(s.length);
    strcpy(data, s.data);
}

String::~String() { cudaFree(data); }

// Assignment operator
String& String::operator=(const char *s) {
    _realloc(strlen(s));
    strcpy(data, s);
    return *this;
}

// Element access (from host or device)
__host__ __device__
char& String::operator[](int pos) { return data[pos]; }

// C-string access
__host__ __device__
const char* String::c_str() const { return data; }

void String::_realloc(int len) {
    cudaFree(data);
    length = len;
    cudaMallocManaged(&data, length + 1);
}
