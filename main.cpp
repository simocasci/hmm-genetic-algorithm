#include "matrix.cpp"
#include "log.cpp"
#include "evolver.cpp"
#include <assert.h>

int main()
{
    const int n_hidden_states = 2;
    const int n_visible_states = 3;
    const int n_observations = 10;
    const int observation_length = 20;
    const int population_size = 100;
    const int n_iterations = 50;
    const double mutation_rate = 0.3;
    const double max_mutation = 0.1;
    const bool verbose = 1;

    std::vector<std::vector<double>> transition_matrix = {
        {0.5, 0.5},
        {0.3, 0.7}};

    assert(transition_matrix.size() == n_hidden_states && transition_matrix[0].size() == n_hidden_states);

    Matrix<double> *transition_probs = new Matrix<double>(transition_matrix);

    std::vector<std::vector<double>> emission_matrix = {
        {0.3, 0.4, 0.3},
        {0.8, 0.1, 0.1}};

    assert(emission_matrix.size() == n_hidden_states && emission_matrix[0].size() == n_visible_states);

    Matrix<double> *emission_probs = new Matrix<double>(emission_matrix);

    std::vector<double> initial_probs = {0.6, 0.4};

    assert(initial_probs.size() == n_hidden_states);

    HMM<double> real_hmm;
    real_hmm.initial_probs = initial_probs;
    real_hmm.transition_probs = transition_probs;
    real_hmm.emission_probs = emission_probs;

    std::cout << "Real transition matrix:" << std::endl;
    Log<double>::print(real_hmm.transition_probs);

    std::cout << std::endl;

    std::cout << "Real emission matrix:" << std::endl;
    Log<double>::print(real_hmm.emission_probs);

    std::cout << std::endl;

    std::cout << "Real initial probabilities:" << std::endl;

    std::cout << std::endl;

    Log<double>::print(real_hmm.initial_probs);

    std::vector<observation> observations(n_observations, observation(observation_length));
    for (int i = 0; i < n_observations; i++)
    {
        observation obs = Evolver<double>::generate_observation(real_hmm, observation_length, n_visible_states);
        observations[i] = obs;
    }

    std::cout << std::endl;

    std::cout << "Observations:" << std::endl;
    for (observation obs : observations)
    {
        Log<int>::print(obs);
    }

    std::cout << std::endl;

    HMM<double> evolved_hmm = Evolver<double>::evolve_hidden_markov_model(observations, population_size, n_iterations, mutation_rate, max_mutation, n_hidden_states, n_visible_states, verbose);

    std::cout << std::endl;

    std::cout << "Evolved transition matrix:" << std::endl;
    Log<double>::print(evolved_hmm.transition_probs);

    std::cout << std::endl;

    std::cout << "Evolved emission matrix:" << std::endl;
    Log<double>::print(evolved_hmm.emission_probs);

    std::cout << std::endl;

    std::cout << "Evolved initial probabilities:" << std::endl;
    Log<double>::print(evolved_hmm.initial_probs);
}