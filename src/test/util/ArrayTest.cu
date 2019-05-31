//
// Created by Peter Rigole on 2019-04-19.
//

#include "ArrayTest.cuh"

/**
 * Must not be in a class, but in the global scope!
 */
__global__
void launchArrayTestDeviceTest(ArrayTest *test) {
    test->deviceTest();
    return;
}


__host__
ArrayTest::ArrayTest() : Test("ArrayTest") {
}

__host__
ArrayTest::~ArrayTest() {
    delete arr;
}

__host__
void ArrayTest::test() {
    hostSetup();
    launchArrayTestDeviceTest<<< 1, 1 >>>(this);
    checkCudaErrors();
    hostTest();
    printf("%s host test successful\n", getName());
    emptyArrayTest();
    setArrayElementTest();
}

__host__
void ArrayTest::hostSetup() {
    arr = new Array<TestContainer>(5);
    TestContainer tc1 = TestContainer(1.0, 2.0, 300.0, 1, 2, 100);
    TestContainer tc2 = TestContainer(4.0, 5.0, 6.0, 4, 5, 6);
    TestContainer tc3 = TestContainer(7.0, 8.0, 9.0, 7, 8, 9);
    arr->append(tc1);
    arr->append(tc2);
    arr->append(tc3);

    cudaDeviceSynchronize();
    checkCudaErrors();
}

__device__
void ArrayTest::deviceTest() {

    float ab = (*arr)[0].addAB();
    assert(ab == 3.0);
    (*arr)[0].setC(ab);

    int xy = (*arr)[2].addXY();
    assert(xy == 15); // should be 15!
    (*arr)[2].setZ(xy);

    float sum = 0;
    for(Array<TestContainer>::iterator i = arr->begin(); i != arr->end(); i++) {
        TestContainer &testContainer = *i;
        sum += testContainer.getC();
    }
    // 3.0 + 6.0 + 9.0 = 18.0 (remember that (*arr)[0]->setC(3.0) above)
    assert(sum  == 18);

    return;
}

__host__
void ArrayTest::emptyArrayTest() {
    empty = new Array<TestContainer>(0);
    cudaDeviceSynchronize();
    checkCudaErrors();
}

__host__
void ArrayTest::setArrayElementTest() {
    Array<TestContainer>* setArray = new Array<TestContainer>(4);
    TestContainer tc1 = TestContainer(1.0, 2.0, 300.0, 1, 2, 100);
    TestContainer tc2 = TestContainer(4.0, 5.0, 6.0, 4, 5, 6);
    TestContainer tc3 = TestContainer(7.0, 8.0, 9.0, 7, 8, 9);
    setArray->set(tc1, 0);
    setArray->set(tc2, 1);
    setArray->set(tc3, 2);
    assert(setArray->getSize() == 3);
}

__host__
void ArrayTest::hostTest() {
    assert((*arr)[0].getC() == 3.0); // should be 3.0!
    assert((*arr)[2].getZ() == 15); // should be 15!
    return;
}
