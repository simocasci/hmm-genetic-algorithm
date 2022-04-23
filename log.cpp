#ifndef LOG_DEF
#define LOG_DEF

#include <iostream>
#include <vector>
#include "matrix.cpp"

template <typename T>
class Log
{
public:
    static void print(const Matrix<T> *m)
    {
        for (int i = 0; i < m->rows; i++)
        {
            std::cout << std::endl;
            for (int j = 0; j < m->cols; j++)
            {
                std::cout << m->data[i][j] << " ";
            }
        }
        std::cout << std::endl;
    }

    static void print(const std::vector<T> &v)
    {
        for (int i = 0; i < v.size(); i++)
        {
            std::cout << v[i] << " ";
        }
        std::cout << std::endl;
    }
};

#endif