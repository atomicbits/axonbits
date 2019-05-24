//
// Created by Peter Rigole on 2019-05-24.
//

#ifndef AXONBITS_TIMER_H
#define AXONBITS_TIMER_H


#include <iostream>
#include <chrono>

// See: https://gist.github.com/gongzhitaao/7062087

class Timer {

public:
    Timer();
    void reset();
    double elapsed() const;

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;

};

#endif //AXONBITS_TIMER_H
