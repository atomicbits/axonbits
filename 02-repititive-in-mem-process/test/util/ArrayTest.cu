//
// Created by Peter Rigole on 2019-04-19.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAYTEST_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAYTEST_H

#include <assert.h>
#include "../../Managed.cuh"
#include "../../util/Array.cu"


class TestContainer : public Managed {

public:

    __device__
    TestContainer() {}

    __device__
    TestContainer(float a_init,
                  float b_init,
                  float c_init,
                  int x_init,
                  int y_init,
                  unsigned int z_init) : a(a_init),
                                         b(b_init),
                                         c(c_init),
                                         x(x_init),
                                         y(y_init),
                                         z(z_init) {}

    __device__
    ~TestContainer() {}

    __device__
    float addAB() { return a + b; }

    __device__
    float addBC() { return b + c; }

    __device__
    int addXY() { return x + y; }

    __device__
    int addYZ() { return y + z; }


private:
    float a;
    float b;
    float c;
    int x;
    int y;
    unsigned int z;

};




class ArrayTest : public Managed {

public:

    __device__
    void test() {
        Array<TestContainer>* arr = new Array<TestContainer>(5);
        arr->append(new TestContainer(1.0, 2.0, 3.0, 1, 2, 3));
        arr->append(new TestContainer(4.0, 5.0, 6.0, 4, 5, 6));
        arr->append(new TestContainer(7.0, 8.0, 9.0, 7, 8, 9));
        assert((*arr)[0].addAB() == 3.0);
        assert((*arr)[1].addBC() == 11.0);
        assert((*arr)[2].addXY() == 15);
        result = 0;
        return;
    }

    int getResult() {
        return result;
    }

private:
    int result = 0;

};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAYTEST_H
