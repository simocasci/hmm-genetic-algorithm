#ifndef HMM_DEF
#define HMM_DEF

#include <vector>
#include "matrix.cpp"

template <typename T>
struct HMM
{
    std::vector<double> initial_probs;
    Matrix<T> *transition_probs;
    Matrix<T> *emission_probs;
};

#endif