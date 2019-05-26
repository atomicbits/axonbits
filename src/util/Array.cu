//
// Created by Peter Rigole on 2019-04-19.
//

#ifndef AXONBITS_ARRAY_H
#define AXONBITS_ARRAY_H

#include "../Managed.cuh"
#include <assert.h>
#include <stdio.h>

// some useful tips:
// https://codereview.stackexchange.com/questions/102036/c-array-with-iterators
// https://en.cppreference.com/w/cpp/iterator/iterator

template<class T>

/**
 * An Array class that runs on host and device, keeps elements and manages their memory on the GPU device.
 *
 * @tparam T
 */
class Array : public Managed {
public:

    __host__
    Array(unsigned int max_size_init) : max_size(max_size_init) {
        if (max_size_init > 0) {
            T *data_init;
            cudaMallocManaged(&data_init, max_size_init * sizeof(T));
            data = data_init;
        }
        // We probably don't want to call in util classes...
//        cudaDeviceSynchronize();
//        cudaError_t cudaError;
//        cudaError = cudaGetLastError();
//        if(cudaError != cudaSuccess) {
//            printf("Device failure during array initialization, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
//        }
    }

    // Destructor
    __host__
    ~Array() {
        if (max_size > 0) {
            cudaFree(data);
        }

        // We probably don't want to call in util classes...
        // cudaDeviceSynchronize();
    }

    /**
     * append is a convenience method to incrementelly fill up an Array.
     * Mind that it doesn't work in combination with set(...), because when using set it is assumed that the
     * array content is managed by the onwer of the array (the size is assumed to be equal to largest index seen while
     * setting the elements).
     */
    __host__
    void append(T &element) {
        if (size == max_size) assert(0); // ToDo: exception handling on device, how? https://stackoverflow.com/questions/50755717/triggering-a-runtime-error-within-a-cuda-kernel ?
        data[size] = element;
        ++size;
        // we're not doing cudaDeviceSynchronize(); here... should we?
    }

//    __host__ __device__
//    void appendMany(T* element, int number) {
//        if (size + number > max_size) assert(0); // ToDo: exception handling on device, how?
//        T* current = element;
//        for(int i = 0; i < number; i++) {
//            data[size] = *current; // not element[i] ???
//            ++current;
//            ++size;
//        }
//    }

    __host__ __device__
    T& operator[](unsigned int index) {
        if(index>=max_size) assert(0); // ToDo: exception handling on device, how?
        return data[index]; // are we returning the address of the element?
    }

//    __host__ __device__
//    const T& operator[](unsigned int index) const {
//        if(index>=max_size) assert(0); // ToDo: exception handling on device, how?
//        return data[index];
//    }

    __host__ __device__
    void set(T &element, unsigned int index) {
        if(index>=max_size) assert(0); // ToDo: exception handling on device, how?
        data[index] = element;
        if(index > size - 1) {
            size = index + 1; // We set the size based on the largest seen index, assuming the owner fills it up without gaps!
        }
    }

    __host__ __device__
    unsigned int getSize() {
        return size;
    }

    // iterator class
    // Mind that the iterator class is not Managed and its instantiation (begin() and end()) is not by the
    // 'new' operator. This is because an iterator is only supposed to be used on thread stack memory!
    //
    // Use:
    // for (Array<T>::iterator i = arr.begin(); i != arr.end(); ++i)
    //    {
    //        T* t = *i;
    //        cout << *i << " ";
    //    }
    class iterator {
        // T* const *data;
        T *data;
    public:

        __host__ __device__
        iterator(T *arr) : data(arr) {}

        __host__ __device__
        T& operator*() {
            return *data;
        }

        __host__ __device__
        const iterator operator++(int) {
            iterator temp = *this;
            ++*this;
            return temp;
        }

        __host__ __device__
        iterator &operator++() {
            ++data;
            return *this;
        }

        __host__ __device__
        friend bool operator==(const iterator& rhs, const iterator& lhs) {
            return rhs.data == lhs.data;
        }

        __host__ __device__
        friend bool operator!=(const iterator& rhs, const iterator& lhs) {
            return !(rhs == lhs);
        }

    };

    __host__ __device__
    iterator begin() const { return iterator(data); }

    __host__ __device__
    iterator end() const{ return iterator(data + size); }

    __host__ __device__
    iterator index(const unsigned int index) const{ return iterator(data + index); }


private:
    T* data;
    unsigned int size = 0;
    unsigned int max_size;
};


#endif //AXONBITS_ARRAY_H
