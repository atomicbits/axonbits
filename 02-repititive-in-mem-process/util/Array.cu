//
// Created by Peter Rigole on 2019-04-19.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAY_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAY_H

#include "../Managed.cuh"
#include <assert.h>

// some useful tips:
// https://codereview.stackexchange.com/questions/102036/c-array-with-iterators
// https://en.cppreference.com/w/cpp/iterator/iterator

template<class T>

class Array : public Managed {
public:

    __host__
    Array(unsigned int max_size_init) : max_size(max_size_init) {
        T** data_init;
        cudaMallocManaged(&data_init, max_size_init * sizeof(T*));
        data = data_init;
        // We don't add cudaDeviceSynchronize(); here because Array is already Managed, so its 'new' will call it.
    }

    // Destructor
    __host__
    ~Array() {
//        for(iterator i = begin(); i != end(); i++ ) {
//            delete i;
//        }
        cudaDeviceSynchronize();
        cudaFree(data);
    }

    __host__ __device__
    void append(T* element) {
        if (size == max_size) assert(0); // ToDo: exception handling on device, how? https://stackoverflow.com/questions/50755717/triggering-a-runtime-error-within-a-cuda-kernel ?
        data[size] = element;
        ++size;
    }

//    __host__ __device__
//    void appendMany(T* element, int number) {
//        if (size + number > max_size) assert(0); // ToDo: exception handling on device, how?
//        T* current = element;
//        for(int i = 0; i < number; i++) {
//            data[size] = current;
//            ++current;
//            ++size;
//        }
//    }

    __host__ __device__
    T* operator[](int index) {
        if(index<0 || index>=size) assert(0); // ToDo: exception handling on device, how?
        return data[index];
    }

//    __host__ __device__
//    const T& operator[](int index) const {
//        if(index<0 || index>=size) assert(0); // ToDo: exception handling on device, how?
//        return data[index];
//    }


    // iterator class
    // Mind that the iterator class is not Managed and its instantiation (begin() and end()) is not by the
    // 'new' operator. This is because an iterator is only supposed to be used on thread stack memory!
    //
    // Use:
    // for (i = arr.begin(); i != arr.end(); ++i)
    //    {
    //        cout << *i << " ";
    //    }
    class iterator {
        const T** data;
    public:

        iterator(const T** arr) : data(arr) {}

        __host__ __device__
        const T* operator*() const {
            return **data;
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
        friend bool operator==(const iterator& rhs,const iterator& lhs) {
            return rhs->data == lhs->data;
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


private:
    T** data; // if you're breaking your head on this, read https://stackoverflow.com/questions/6130712/pointer-to-array-of-pointers
    unsigned int size = 0;
    unsigned int max_size;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAY_H
