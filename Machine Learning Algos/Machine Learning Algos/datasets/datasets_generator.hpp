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
	// sampling from a stanadard normal dist
	if (random_state != INT_MAX) arma_rng::set_seed(random_state);
	mat x(n_samples, n_features, fill::randn);
	// generate a random y
	vec *ground_truth = new vec(n_features, fill::randu);
	vec y = x * (*ground_truth) * 100 + bias;
	delete ground_truth;
	if (noise > 0.0) y += vec(n_samples, fill::randn) * noise;
	return tuple<mat, vec>(x, y);
}


// make classification problem
tuple<mat, vec> make_classification(
	int n_samples = 100,
	int n_features = 2,
	int n_classes = 2,
	double noise = 0.0,
	int random_state = INT_MAX
) {
	// sampling from a stanadard normal dist and init y
	if (random_state != INT_MAX) arma_rng::set_seed(random_state);
	mat x(n_samples, n_features, fill::randn);
	vec y(n_samples, fill::zeros);

	// get number of samples for each class
	uvec n_samples_perk(n_classes);
	n_samples_perk.fill(u64(n_samples / n_classes));
	
	for (int i = 0; i < n_samples - sum(n_samples_perk); ++i) {
		n_samples_perk(i) += 1;
	}

	// put y values
	int stop = 0, start;
	for (int i = 0; i < n_classes; ++i) {
		start = stop;
		stop += int(n_samples_perk(i));
		y.subvec(start, stop -1).fill(i);
	}
	return tuple<mat, vec>(x, y);
}

// dataloader
tuple<mat, vec> dataloader(string filename) {
	string file = "datasets/load_data/" + filename + ".csv";
	mat data;
	data.load(file);

	mat X = data.cols(0, data.n_cols - 2);
	vec y = data.col(data.n_cols - 1);

	return tuple<mat, vec>(X, y);
}