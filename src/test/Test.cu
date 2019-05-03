
//
// Created by Peter Rigole on 2019-04-24.
//

#include "Test.cuh"
#include "util/ArrayTest.cu"

Test::Test() : type(TestClass::unknown) {}

Test::Test(const TestClass type_init): type(type_init) {}

Test::~Test() {}

__host__
void Test::hostTest() {
    if(type == TestClass::arraytest) {
        static_cast<ArrayTest*>(this)->hostTest();
    } else {
        assert(0);
    }
}

__host__
const char* Test::getName() {
    if(type == TestClass::arraytest) {
        return static_cast<ArrayTest*>(this)->getName();
    } else {
        assert(0);
        return "";
    }
}

__device__
void Test::deviceTest() {
    if(type == TestClass::arraytest) {
        static_cast<ArrayTest*>(this)->deviceTest();
    } else {
        assert(0);
    }
}
