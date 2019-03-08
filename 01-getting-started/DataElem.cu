
#include <cuda_runtime.h>

#include <stdio.h>

#include "Managed.cuh"
#include "String.cuh"
#include "DataElement.cuh"


__global__
void Kernel_by_pointer(DataElement *elem) {
    printf("On device by pointer:       name=%s, value=%d\n", elem->name.c_str(), elem->value);

    elem->name[0] = 'p';
    elem->value++;
}

__global__
void Kernel_by_ref(DataElement &elem) {
    printf("On device by ref:           name=%s, value=%d\n", elem.name.c_str(), elem.value);

    elem.name[0] = 'r';
    elem.value++;
}

__global__
void Kernel_by_value(DataElement elem) {
    printf("On device by value:         name=%s, value=%d\n", elem.name.c_str(), elem.value);

    elem.name[0] = 'v';
    elem.value++;
}

void launch_by_pointer(DataElement *elem) {
    Kernel_by_pointer<<< 1, 1 >>>(elem);
    cudaDeviceSynchronize();
}

void launch_by_ref(DataElement &elem) {
    Kernel_by_ref<<< 1, 1 >>>(elem);
    cudaDeviceSynchronize();
}

void launch_by_value(DataElement elem) {
    Kernel_by_value<<< 1, 1 >>>(elem);
    cudaDeviceSynchronize();
}



int main(int argc, char **argv)
{

    DataElement *e = new DataElement;

    e->value = 10;
    e->name = "hello";

    launch_by_pointer(e);

    printf("On host (after by-pointer): name=%s, value=%d\n", e->name.c_str(), e->value);

    launch_by_ref(*e);

    printf("On host (after by-ref):     name=%s, value=%d\n", e->name.c_str(), e->value);

    launch_by_value(*e);

    printf("On host (after by-value):   name=%s, value=%d\n", e->name.c_str(), e->value);

    //delete e;

    cudaDeviceReset();

}
