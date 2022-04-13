#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <random>

#define N_HIDDEN_STATES 2
#define N_VISIBLE_STATES 3
#define VERBOSE 1

typedef std::vector<std::vector<double>> matrix;
typedef std::vector<int> observation;

struct HiddenMarkovModel {
    matrix transition;
    matrix emission;
    std::vector<double> initial;
};

double random_range(int left, int right) {
    std::uniform_real_distribution<double> dist(left, right);
    std::mt19937 rng; 
    rng.seed(std::random_device{}()); 
    return dist(rng);
}

int random_choice(std::vector<double> probabilities) {
    int n = probabilities.size();
    double r = random_range(0, 1);
    double s = 0;

    for (int i = 0; i < n-1; i++) {
        s += probabilities[i];
        if (s >= r) return i;
    }

    return n - 1;
}

void print_matrix(matrix& m) {
    int rows = m.size();
    int cols = m[0].size();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_vector(std::vector<double>& v) {
    int n = v.size();

    for (int i = 0; i < n; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

void print_observation(observation obs) {
    int n = obs.size();

    for (int i = 0; i < n; i++) {
        std::cout << obs[i] << " ";
    }
    std::cout << std::endl;
}

void normalize_rows(matrix& m) {
    int rows = m.size();
    int cols = m[0].size();

    std::vector<double> rows_sums(rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (m[i][j] < 0) m[i][j] = 0;
            rows_sums[i] += m[i][j];
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m[i][j] /= rows_sums[i];
        }
    }
}

void normalize_vector(std::vector<double>& v) {
    int n = v.size();
    double total = 0;

    for (int i = 0; i < n; i++) {
        if (v[i] < 0) v[i] = 0;
        total += v[i];
    }

    for (int i = 0; i < n; i++) {
        v[i] /= total;
    }
}

matrix generate_random_prob_matrix(int rows, int cols) {
    std::vector<std::vector<double>> random_prob_matrix(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            random_prob_matrix[i][j] = random_range(0, 1);
        }
    }

    normalize_rows(random_prob_matrix);

    return random_prob_matrix;
}

std::vector<double> generate_random_prob_vector(int n) {
    std::vector<double> random_prob_vector(n);

    for (int i = 0; i < n; i++) {
        random_prob_vector[i] = random_range(0, 1);
    }

    normalize_vector(random_prob_vector);

    return random_prob_vector;
}

void mutate_matrix(matrix& prob_matrix, double max_mutation) {
    int rows = prob_matrix.size();
    int cols = prob_matrix[0].size();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            prob_matrix[i][j] += random_range(-max_mutation, max_mutation);
        }
    }

    normalize_rows(prob_matrix);
}

void mutate_vector(std::vector<double>& v, double max_mutation) {
    int n = v.size();

    std::uniform_real_distribution<double> dist(-max_mutation, max_mutation);
    std::mt19937 rng; 
    rng.seed(std::random_device{}());

    for (int i = 0; i < n; i++) {
        v[i] += dist(rng);
    }

    normalize_vector(v);
}

matrix cross_matrices(matrix& first, matrix& second) {
    int rows = first.size();
    int cols = first[0].size();

    matrix crossed(rows, std::vector<double>(cols));

    for (int i = 0; i <= rows / 2; i++) {
        for (int j = 0; j < cols; j++) {
            if (crossed[i][j] == 0) crossed[i][j] = first[i][j];
            if ((rows / 2) + i < rows) crossed[(rows / 2) + i][j] = second[(rows / 2) + i][j];
        }
    }
    
    return crossed;
}

std::vector<double> cross_vectors(std::vector<double> v1, std::vector<double> v2) {
    int n = v1.size();
    std::vector<double> crossed(n);

    for (int i = 0; i <= n / 2; i++) {
        if (crossed[i] == 0) crossed[i] = v1[i];
        if ((n / 2) + i < n) crossed[(n / 2) + i] = v2[(n / 2) + i];
    }

    return crossed;
}

std::vector<matrix> generate_random_matrix_population(int rows, int cols, int population_size) {
    std::vector<matrix> random_population(population_size);
    for (int i = 0; i < population_size; i++) {
        matrix random_matrix = generate_random_prob_matrix(rows, cols);
        random_population[i] = random_matrix;
    }
    return random_population;
}

std::vector<std::vector<double>> generate_random_vector_population(int n, int population_size) {
    std::vector<std::vector<double>> random_population(population_size);
    for (int i = 0; i < population_size; i++) {
        std::vector<double> random_vector = generate_random_prob_vector(n);
        random_population[i] = random_vector;
    }
    return random_population;
}

observation generate_observation(HiddenMarkovModel hmm, int observation_length) {
    observation obs = observation(observation_length);
    int current_visible_state = 0;
    int current_hidden_state = 0;

    for (int i = 0; i < observation_length; i++) {
        if (i == 0) {
            current_hidden_state = random_choice(hmm.initial);
            std::vector<double> visible_states_initial(N_VISIBLE_STATES, 1.0 / N_VISIBLE_STATES);
            current_visible_state = random_choice(visible_states_initial);
        }
        else {
            current_hidden_state = random_choice(hmm.transition[current_hidden_state]);
            current_visible_state = random_choice(hmm.emission[current_hidden_state]);
        }

        obs[i] = current_visible_state;
    }

    return obs;
}

int score_prediction(observation predicted_obs, observation actual_obs) {
    int n = predicted_obs.size();
    int score = 0;

    for (int i = 0; i < n; i++) {
        score += predicted_obs[i] == actual_obs[i];
    }

    return score;
}

std::vector<double> get_population_scores(std::vector<HiddenMarkovModel>& population, std::vector<observation>& observations) {
    std::vector<double> scores(population.size());
    for (int i = 0; i < population.size(); i++) {
        int total_score = 0;
        for (observation obs: observations) {
            observation predicted_obs = generate_observation(population[i], observations[0].size());
            int score = score_prediction(predicted_obs, obs);
            total_score += score;
        }
        scores[i] = ((double)total_score) / observations.size();
    }
    return scores;
}

HiddenMarkovModel crossover(HiddenMarkovModel& first, HiddenMarkovModel& second) {
    HiddenMarkovModel result;
    result.transition = cross_matrices(first.transition, second.transition);
    result.emission = cross_matrices(first.emission, second.emission);
    result.initial = cross_vectors(first.initial, second.initial);
    return result;
}

void mutate(HiddenMarkovModel& hmm, double max_mutation) {
    mutate_matrix(hmm.transition, max_mutation);
    mutate_matrix(hmm.emission, max_mutation);
    mutate_vector(hmm.initial, max_mutation);
}

HiddenMarkovModel evolve_hidden_markov_model(std::vector<observation>& observations, int population_size, int n_iterations, double mutation_rate, double max_mutation) {
    HiddenMarkovModel best;

    std::vector<matrix> transition_population = generate_random_matrix_population(N_HIDDEN_STATES, N_HIDDEN_STATES, population_size);
    std::vector<matrix> emission_population = generate_random_matrix_population(N_HIDDEN_STATES, N_VISIBLE_STATES, population_size);
    std::vector<std::vector<double>> initial_population = generate_random_vector_population(N_HIDDEN_STATES, population_size);
    
    std::vector<HiddenMarkovModel> population(population_size);
    
    for (int i = 0; i < population_size; i++) {
        HiddenMarkovModel hmm = {transition_population[i], emission_population[i], initial_population[i]};
        population[i] = hmm;
    }
    
    for (int iteration = 0; iteration < n_iterations; iteration++) {
        if (VERBOSE) std::cout << "Iteration: " << iteration << " - ";

        std::vector<double> scores = get_population_scores(population, observations);
        double total_score = 0;
        for (int i = 0; i < scores.size(); i++) {
            total_score += scores[i];
        }

        if (iteration < n_iterations-1) {
            double best_score = 0;
            std::vector<HiddenMarkovModel> new_population(population_size);

            std::vector<double> normalized_scores(scores.size());
            for (int i = 0; i < scores.size(); i++) {
                normalized_scores[i] = scores[i] / total_score;
                if (scores[i] > best_score) best_score = scores[i];
            }

            if (VERBOSE) std::cout << "Best score: " << best_score << std::endl;

            for (int i = 0; i < population_size; i++) {
                HiddenMarkovModel first_parent = population[random_choice(normalized_scores)];
                HiddenMarkovModel second_parent = population[random_choice(normalized_scores)];
                HiddenMarkovModel child = crossover(first_parent, second_parent);                
                if (random_range(0, 1) <= mutation_rate) mutate(child, max_mutation);
                new_population[i] = child;
            }
            population = new_population;
        } else {
            double best_score = 0;
            for (int i = 0; i < population_size; i++) {
                if (scores[i] > best_score) {
                    best = population[i];
                    best_score = scores[i];
                }
            }
            if (VERBOSE) std::cout << "Best score: " << best_score << std::endl;
        }
    }

    return best;
}

int main() {
    matrix transition = {
        {0.5, 0.5},
        {0.3, 0.7}
    };

    assert(transition.size() == N_HIDDEN_STATES && transition[0].size() == N_HIDDEN_STATES);

    matrix emission = {
        {0.3, 0.4, 0.3},
        {0.8, 0.1, 0.1}
    };

    assert(emission.size() == N_HIDDEN_STATES && emission[0].size() == N_VISIBLE_STATES);

    std::vector<double> initial = {0.6, 0.4};

    assert(initial.size() == N_HIDDEN_STATES);

    HiddenMarkovModel real_hmm = {transition, emission, initial};

    std::cout << "Real transition matrix:" << std::endl;
    print_matrix(real_hmm.transition);

    std::cout << "Real emission matrix:" << std::endl;
    print_matrix(real_hmm.emission);

    std::cout << "Real initial probabilities:" << std::endl;
    print_vector(real_hmm.initial);

    const int n_observations = 10;
    const int observation_length = 20;
    const int population_size = 100;
    const int n_iterations = 50;
    const double mutation_rate = 0.3;
    const double max_mutation = 0.1;

    std::vector<observation> observations(n_observations, observation(observation_length));
    for (int i = 0; i < n_observations; i++) {
        observation obs = generate_observation(real_hmm, observation_length);
        observations[i] = obs;
    }

    std::cout << std::endl;

    std::cout << "Observations:" << std::endl;
    for (observation obs: observations) {
        print_observation(obs);
    }

    std::cout << std::endl;

    HiddenMarkovModel evolved_hmm = evolve_hidden_markov_model(observations, population_size, n_iterations, mutation_rate, max_mutation);

    std::cout << std::endl;

    std::cout << "Evolved transition matrix:" << std::endl;
    print_matrix(evolved_hmm.transition);

    std::cout << "Evolved emission matrix:" << std::endl;
    print_matrix(evolved_hmm.emission);

    std::cout << "Evolved initial probabilities:" << std::endl;
    print_vector(evolved_hmm.initial);
}