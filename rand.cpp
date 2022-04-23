#ifndef RAND_DEF
#define RAND_DEF

#include <random>
#include <vector>

class Rand
{
public:
    static double range(const int &&left, const int &&right)
    {
        std::uniform_real_distribution<double> dist(left, right);
        std::mt19937 random_number_generator;
        random_number_generator.seed(std::random_device{}());
        return dist(random_number_generator);
    }

    static int choice(const std::vector<double> &probs)
    {
        int n = probs.size();
        double r = Rand::range(0, 1);
        double s = 0;

        for (int i = 0; i < n - 1; i++)
        {
            s += probs[i];
            if (s >= r)
                return i;
        }

        return n - 1;
    }
};

#endif