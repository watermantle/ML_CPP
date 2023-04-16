/*
Header file of datasets with some functions to generate datasets for various
purposes.


*/

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;


pair<Eigen::MatrixXd, Eigen::VectorXd> make_regression(
	const int n_samples = 100,
	const int n_features = 100,
	const double bias = 0.0,
	const double noise = 0.0,
	const int random_state = INT_MAX
) {
	// sampling from a stanadard normal dist
	if (random_state != INT_MAX) srand(random_state);
	Eigen::MatrixXd x = Eigen::MatrixXd::Random(n_samples, n_features);
	// generate a random y
	Eigen::VectorXd ground_truth = Eigen::VectorXd::Random(n_features);
	Eigen::VectorXd y = 100 * x * ground_truth + Eigen::VectorXd::Constant(n_samples, bias);

	if (noise > 0.0) y += Eigen::VectorXd::Random(n_samples) * noise;
	return make_pair(x, y);
}


// make classification problem
pair<Eigen::MatrixXd, Eigen::VectorXd> make_classification(
	int n_samples = 100,
	int n_features = 2,
	int n_classes = 2,
	double noise = 0.0,
	int random_state = INT_MAX
) {
	// sampling from a stanadard normal dist and init y
	if (random_state != INT_MAX) srand(random_state);
	Eigen::MatrixXd x = Eigen::MatrixXd::Random(n_samples, n_features);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(n_samples);

	// get number of samples for each class
	VectorXi n_samples_perk(n_classes);
	n_samples_perk.fill(n_samples / n_classes);
	
	for (int i = 0; i < n_samples - n_samples_perk.array().sum(); ++i) {
		n_samples_perk(i)++;
	}

	// put y values
	int stop = 0, start;
	for (int i = 0; i < n_classes; ++i) {
		start = stop;
		stop += int(n_samples_perk(i));
		y.segment(start, stop - start).setConstant(i);
	}
	return make_pair(x, y);
}

// dataloader
pair<Eigen::MatrixXd, Eigen::VectorXd> dataloader(string filename) {
	// Construct the file path
	string file = "datasets/load_data/" + filename + ".csv";

	// Open the file
	ifstream data_file(file);
	if (!data_file.is_open()) {
		throw std::runtime_error("Cannot open file: " + file);
	}

	// Read the data from the file
	vector<double> data_buffer;
	string line;
	int rows = 0;
	int cols = 0;

	// Read each line and split by comma
	while (getline(data_file, line)) {
		stringstream line_stream(line);
		string cell;
		cols = 0;

		// Read each cell separated by a comma
		while (getline(line_stream, cell, ',')) {
			data_buffer.push_back(stod(cell));
			cols++;
		}
		rows++;
	}

	// Create the Eigen::MatrixXd object from the read data
	Eigen::MatrixXd data(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			data(i, j) = data_buffer[i * cols + j];
		}
	}

	// Separate the last column as the target vector
	Eigen::MatrixXd X = data.leftCols(data.cols() - 1);
	Eigen::VectorXd y = data.col(data.cols() - 1);

	// Return the pair of Eigen::MatrixXd and Eigen::VectorXd objects
	return make_pair(X, y);
}