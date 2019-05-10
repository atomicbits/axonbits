//
// Created by Peter Rigole on 2019-05-10.
//

#ifndef AXONBITS_TESTCONTAINER_H
#define AXONBITS_TESTCONTAINER_H

class TestContainer {

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


#endif //AXONBITS_TESTCONTAINER_H
