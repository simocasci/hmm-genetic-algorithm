#ifndef EVOLVER_DEF
#define EVOLVER_DEF

#include <vector>
#include <iostream>
#include "hmm.h"
#include "matrix.cpp"
#include "utils.cpp"
#include "rand.cpp"

typedef std::vector<int> observation;

template <typename T>
class Evolver
{
public:
    static void mutate_matrix(Matrix<T> *matrix, const double &max_mutation)
    {
        for (int i = 0; i < matrix->rows; i++)
        {
            for (int j = 0; j < matrix->cols; j++)
            {
                matrix->data[i][j] += Rand::range(-max_mutation, max_mutation);
            }
        }

        Matrix<T>::normalize(matrix);
    }

    static Matrix<T> *cross_matrices(const Matrix<T> *first, const Matrix<T> *second)
    {
        const int &rows = first->rows;
        const int &cols = first->cols;

        std::vector<std::vector<T>> crossed_data(rows, std::vector<T>(cols));

        for (int i = 0; i <= rows / 2; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (crossed_data[i][j] == 0)
                    crossed_data[i][j] = first->data[i][j];
                if ((rows / 2) + i < rows)
                    crossed_data[(rows / 2) + i][j] = second->data[(rows / 2) + i][j];
            }
        }

        Matrix<T> *crossed_matrix = new Matrix<T>(crossed_data);
        return crossed_matrix;
    }

    static observation generate_observation(const HMM<double> &hmm, const int &observation_length, const int &n_visible_states)
    {
        observation obs(observation_length);

        int current_visible_state = 0;
        int current_hidden_state = 0;

        current_hidden_state = Rand::choice(hmm.initial_probs);
        std::vector<double> visible_states_initial(n_visible_states, 1.0 / n_visible_states);
        current_visible_state = Rand::choice(visible_states_initial);

        obs[0] = current_visible_state;

        for (int i = 1; i < observation_length; i++)
        {
            current_hidden_state = Rand::choice((hmm.transition_probs)->data[current_hidden_state]);
            current_visible_state = Rand::choice((hmm.emission_probs)->data[current_hidden_state]);

            obs[i] = current_visible_state;
        }

        return obs;
    }

    static HMM<double> crossover(const HMM<double> &first, const HMM<double> &second)
    {
        HMM<double> result;

        result.initial_probs = cross_vectors(first.initial_probs, second.initial_probs);
        result.transition_probs = Evolver<T>::cross_matrices(first.transition_probs, second.transition_probs);
        result.emission_probs = Evolver<T>::cross_matrices(first.emission_probs, second.emission_probs);

        return result;
    }

    static void mutate(HMM<double> &hmm, const double &max_mutation)
    {
        mutate_vector(hmm.initial_probs, max_mutation);
        Evolver<T>::mutate_matrix(hmm.transition_probs, max_mutation);
        Evolver<T>::mutate_matrix(hmm.emission_probs, max_mutation);
    }

    static std::vector<double> get_population_scores(const std::vector<HMM<double>> &population, const std::vector<observation> &observations, const int &n_visible_states)
    {
        std::vector<double> scores(population.size());

        for (int i = 0; i < population.size(); i++)
        {
            int total_score = 0;
            for (observation obs : observations)
            {
                observation predicted_obs = Evolver<T>::generate_observation(population[i], observations[0].size(), n_visible_states);
                int score = Evolver<T>::score_prediction(predicted_obs, obs);
                total_score += score;
            }

            scores[i] = ((double)total_score) / observations.size();
        }

        return scores;
    }

    static std::vector<std::vector<double>> generate_random_vector_population(const int &vector_size, const int &population_size)
    {
        std::vector<std::vector<double>> random_population(population_size);

        for (int i = 0; i < population_size; i++)
        {
            std::vector<double> random_vector = generate_rand_prob_vec(vector_size);
            random_population[i] = random_vector;
        }

        return random_population;
    }

    static int score_prediction(const observation &predicted_obs, const observation &actual_obs)
    {
        int score = 0;

        for (int i = 0; i < predicted_obs.size(); i++)
        {
            score += predicted_obs[i] == actual_obs[i];
        }

        return score;
    }

    static HMM<T> evolve_hidden_markov_model(const std::vector<observation> &observations, const int &population_size, const int &n_iterations, const double &mutation_rate, const double &max_mutation, const int &n_hidden_states, const int &n_visible_states, const bool &verbose)
    {
        HMM<T> best;

        std::vector<HMM<T>> population(population_size);

        for (int i = 0; i < population_size; i++)
        {
            std::vector<double> initial_probs = generate_rand_prob_vec(n_hidden_states);
            Matrix<T> *transition_probs = new Matrix<T>(n_hidden_states, n_hidden_states);
            Matrix<T> *emission_probs = new Matrix<T>(n_hidden_states, n_visible_states);

            HMM<T> hmm;
            hmm.initial_probs = initial_probs;
            hmm.transition_probs = transition_probs;
            hmm.emission_probs = emission_probs;

            population[i] = hmm;
        }

        for (int iteration = 0; iteration < n_iterations; iteration++)
        {
            if (verbose)
                std::cout << "Iteration: " << iteration << " - ";

            std::vector<double> scores = get_population_scores(population, observations, n_visible_states);

            double total_score = 0;

            for (int i = 0; i < scores.size(); i++)
            {
                total_score += scores[i];
            }

            if (iteration < n_iterations - 1)
            {
                double best_score = 0;

                std::vector<HMM<T>> new_population(population_size);

                std::vector<double> normalized_scores(scores.size());

                for (int i = 0; i < scores.size(); i++)
                {
                    normalized_scores[i] = scores[i] / total_score;
                    if (scores[i] > best_score)
                        best_score = scores[i];
                }

                if (verbose)
                    std::cout << "Best score: " << best_score << std::endl;

                for (int i = 0; i < population_size; i++)
                {
                    HMM<T> first_parent = population[Rand::choice(normalized_scores)];
                    HMM<T> second_parent = population[Rand::choice(normalized_scores)];
                    HMM<T> child = Evolver<T>::crossover(first_parent, second_parent);

                    if (Rand::range(0, 1) <= mutation_rate)
                        Evolver<T>::mutate(child, max_mutation);
                    new_population[i] = child;
                }
                population = new_population;
            }
            else
            {
                double best_score = 0;
                for (int i = 0; i < population_size; i++)
                {
                    if (scores[i] > best_score)
                    {
                        best.initial_probs = population[i].initial_probs;
                        best.transition_probs = population[i].transition_probs;
                        best.emission_probs = population[i].emission_probs;

                        best_score = scores[i];
                    }
                }
                if (verbose)
                    std::cout << "Best score: " << best_score << std::endl;
            }
        }

        for (int i = 0; i < population_size; i++)
        {
            HMM<T> hmm = population[i];
            if (hmm.transition_probs != best.transition_probs && hmm.emission_probs != best.emission_probs)
            {
                delete hmm.transition_probs;
                delete hmm.emission_probs;
            }
        }

        return best;
    }

    ~Evolver() {}
};

#endif