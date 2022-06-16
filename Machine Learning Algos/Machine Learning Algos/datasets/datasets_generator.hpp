/*
Header file of datasets with some functions to generate datasets for various
purposes.


*/

#include <iostream>
#include <tuple>
#include <armadillo>

using namespace std;
using namespace arma;


tuple<mat, vec> make_regression(
	int n_samples = 100,
	int n_features = 100,
	double bias = 0.0,
	double noise = 0.0,
	int random_state = INT_MAX
) {
	if (random_state != INT_MAX) arma_rng::set_seed(random_state);
	mat X(n_samples, n_features, fill::randn);
	// generate a random y
	vec *ground_truth = new vec(n_features, fill::randu);
	vec y = X * (*ground_truth) * 100 + bias;
	delete ground_truth;
	if (noise > 0.0) y += vec(n_samples, fill::randn) * noise;
	return tuple<mat, vec>(X, y);
}