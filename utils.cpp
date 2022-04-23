#ifndef UTILS_DEF
#define UTILS_DEF

#include <vector>
#include "rand.cpp"
#include "matrix.cpp"
#include "hmm.h"
#include "evolver.cpp"

std::vector<double> generate_rand_prob_vec(const int &n)
{
    std::vector<double> random_prob_vector(n);

    for (int i = 0; i < n; i++)
    {
        random_prob_vector[i] = Rand::range(0, 1);
    }

    Matrix<double>::normalize(random_prob_vector);

    return random_prob_vector;
}

void normalize_vector(std::vector<double> &v)
{
    double total = 0;

    for (int i = 0; i < v.size(); i++)
    {
        if (v[i] < 0)
            v[i] = 0;
        total += v[i];
    }

    for (int i = 0; i < v.size(); i++)
    {
        v[i] /= total;
    }
}

void mutate_vector(std::vector<double> &v, const double &max_mutation)
{
    for (int i = 0; i < v.size(); i++)
    {
        v[i] += Rand::range(-max_mutation, max_mutation);
    }

    Matrix<double>::normalize(v);
}

std::vector<double> cross_vectors(const std::vector<double> &v1, const std::vector<double> &v2)
{
    int n = v1.size();
    std::vector<double> crossed(n);

    for (int i = 0; i <= n / 2; i++)
    {
        if (crossed[i] == 0)
            crossed[i] = v1[i];
        if ((n / 2) + i < n)
            crossed[(n / 2) + i] = v2[(n / 2) + i];
    }

    return crossed;
}

#endif