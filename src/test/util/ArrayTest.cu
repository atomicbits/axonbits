//
// Created by Peter Rigole on 2019-04-19.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAYTEST_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAYTEST_H

#include <assert.h>
#include "../../Managed.cuh"
#include "../../util/Array.cu"
#include "../Test.cuh"


class TestContainer : public Managed {

public:

    __host__
    TestContainer() {}

    __host__
    TestContainer(float a_init,
                  float b_init,
                  float c_init,
                  int x_init,
                  int y_init,
                  int z_init) : a(a_init),
                                         b(b_init),
                                         c(c_init),
                                         x(x_init),
                                         y(y_init),
                                         z(z_init) {}

    __host__
    ~TestContainer() {}

    __device__
    float addAB() { return a + b; }

    __host__ __device__
    float getC() { return c; }

    __device__
    void setC(float c_upd) { c = c_upd; }

    __device__
    int addXY() { return x + y; }

    __host__ __device__
    int getZ() { return z; }

    __device__
    void setZ(int z_upd) { z = z_upd; }


private:
    float a;
    float b;
    float c;
    int x;
    int y;
    int z;

};



class ArrayTest : public Managed, public Test {

public:

    // inline static const string TYPE = "ArrayTest";

    __host__
    ArrayTest() : Test(TestClass::arraytest) {
        arr = new Array<TestContainer>(5);
        arr->append(new TestContainer(1.0, 2.0, 300.0, 1, 2, 100));
        arr->append(new TestContainer(4.0, 5.0, 6.0, 4, 5, 6));
        arr->append(new TestContainer(7.0, 8.0, 9.0, 7, 8, 9));
    }

    __host__
    ~ArrayTest() {
        delete arr;
    }

    __host__
    const char* getName() {
        return "ArrayTest";
    }

    __device__
    void deviceTest() {

        float ab = (*arr)[0]->addAB();
        assert(ab == 3.0);
        (*arr)[0]->setC(ab);

        int xy = (*arr)[2]->addXY();
        assert(xy == 15); // should be 15!
        (*arr)[2]->setZ(xy);

        float sum = 0;
        for(Array<TestContainer>::iterator i = arr->begin(); i != arr->end(); i++) {
            TestContainer* testContainer = *i;
            sum += testContainer->getC();
        }
        // 3.0 + 6.0 + 9.0 = 18.0 (remember that (*arr)[0]->setC(3.0) above)
        assert(sum  == 18);

        return;
    }

    __host__
    void hostTest() {
        assert((*arr)[0]->getC() == 3.0); // should be 3.0!
        assert((*arr)[2]->getZ() == 15); // should be 15!
        return;
    }

    __host__
    Array<TestContainer>* getArray() {
        return arr;
    }

private:
    Array<TestContainer>* arr;

};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_ARRAYTEST_H
