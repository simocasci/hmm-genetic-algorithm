#ifndef MATRIX_DEF
#define MATRIX_DEF

#include <vector>
#include "rand.cpp"

template <typename T>
class Matrix
{
private:
    static std::vector<std::vector<T>> generate_random_data(const int &rows, const int &cols)
    {
        std::vector<std::vector<T>> random_data(rows, std::vector<T>(cols));

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                random_data[i][j] = Rand::range(0, 1);
            }
        }

        return random_data;
    }

public:
    const int rows;
    const int cols;
    std::vector<std::vector<T>> data;

    Matrix(const int &r, const int &c) : rows(r), cols(c), data(Matrix<T>::generate_random_data(rows, cols))
    {
        Matrix<T>::normalize(this);
    }

    Matrix(std::vector<std::vector<T>> &matrix_data) : rows(matrix_data.size()), cols(matrix_data[0].size()), data(matrix_data) {}

    static void normalize(Matrix<T> *m)
    {
        std::vector<T> rows_sums(m->rows);

        for (int i = 0; i < m->rows; i++)
        {
            for (int j = 0; j < m->cols; j++)
            {
                if (m->data[i][j] < 0)
                    m->data[i][j] = 0;

                rows_sums[i] += m->data[i][j];
            }
        }

        for (int i = 0; i < m->rows; i++)
        {
            for (int j = 0; j < m->cols; j++)
            {
                m->data[i][j] /= rows_sums[i];
            }
        }
    }

    static void normalize(std::vector<T> &v)
    {
        T total = 0;

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

    ~Matrix() {}
};

#endif