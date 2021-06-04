/*
source file to apply LinearModels header file
*/

#include <iostream>
#include <armadillo>
#include <limits>
#include "LinearModels.hpp"
using namespace std;
using namespace arma;

// 1. Linear Regression
//	constructors and destructors
LinearRegression::LinearRegression() : fit_intercept(true) {};
LinearRegression::LinearRegression(const LinearRegression& source) : fit_intercept(source.fit_intercept), beta(source.beta) {};
LinearRegression::LinearRegression(bool fit_intercept1) : fit_intercept(fit_intercept1) {};
LinearRegression::~LinearRegression() {};

//	assignment operation
LinearRegression& LinearRegression::operator=(const LinearRegression& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		fit_intercept = source.fit_intercept;
		beta = source.beta;
	}
	return *this;
}

//	functions
// fit function to calculate the results and store to beta
const void LinearRegression::fit(mat X, const mat& y) {
	//	add an addtional column if fit intercept
	if (fit_intercept == true) X = join_rows(ones(X.n_rows), X);
	//	ipseduo inverse part of the equation
	mat ipseudo_inverse = (X.t() * X).i() * X.t();

	beta = ipseudo_inverse * y;		//	save results
	return;
}

// return predicted values with trained model
const mat LinearRegression::predict(mat _X) {
	//	_X is a test matrix with shape of (Z, M), pred_y will be the product of _X and beta
	if (fit_intercept == true) _X = join_rows(ones(_X.n_rows), _X);
	
	mat y_pred = _X * beta;
	return y_pred;
}


//	2. Ridge Regression
RidgeRegression::RidgeRegression() : fit_intercept(true), alpha(1.0) {};
RidgeRegression::RidgeRegression(const RidgeRegression& source) : fit_intercept(source.fit_intercept), beta(source.beta), alpha(source.alpha) {};
RidgeRegression::RidgeRegression(double alpha1, bool fit_intercept1) : fit_intercept(fit_intercept1), alpha(alpha1) {};
RidgeRegression::~RidgeRegression() {};

//	assignment operation
RidgeRegression& RidgeRegression::operator=(const RidgeRegression& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		fit_intercept = source.fit_intercept;
		alpha = source.alpha;
		beta = source.beta;
	}
	return *this;
}


//	functions
// fit function to calculate the results and store to beta
const void RidgeRegression::fit(mat X, const mat& y) {
	//	add an addtional column if fit intercept
	if (fit_intercept == true) X = join_rows(ones(X.n_rows), X);

	//	matrix for alpha

	mat A = alpha * eye(X.n_cols, X.n_cols);
	//	ipseduo inverse part of the equation
	mat ipseudo_inverse = (X.t() * X + A).i() * X.t();

	beta = ipseudo_inverse * y;		//	save results
	return;
}

// return predicted values with trained model
const mat RidgeRegression::predict(mat _X) {
	//	_X is a test matrix with shape of (Z, M), pred_y will be the product of _X and beta
	if (fit_intercept == true) _X = join_rows(ones(_X.n_rows), _X);
	mat y_pred = _X * beta;

	return y_pred;
}


// 3. Logistic Regression
//	constructors and destructors
LogisticRegression::LogisticRegression() : fit_intercept(true), gamma(0.0), penalty("l2") {};
LogisticRegression::LogisticRegression(const LogisticRegression& source) : fit_intercept(source.fit_intercept), beta(source.beta), gamma(source.gamma), penalty(source.penalty) {};
LogisticRegression::LogisticRegression(double gamma1, bool fit_intercept1, string penalty1) : fit_intercept(fit_intercept1), penalty(penalty1), gamma(gamma1) {};
LogisticRegression::~LogisticRegression() {};

//	assignment operation
LogisticRegression& LogisticRegression::operator=(const LogisticRegression& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		fit_intercept = source.fit_intercept;
		gamma = source.gamma;
		penalty = source.penalty;
		beta = source.beta;
	}
	return *this;
}


//	functions
// fit function to calculate the results and store to beta
const void LogisticRegression::fit(mat _X, const mat& y, double lr, double tol, long max_iter) {
	//	whether to fit an intercept
	if (fit_intercept == true) _X = join_rows(ones(_X.n_rows), _X);
	//	previous loss value to store loss
	double loss, l_prev = 999999.0;
	beta = vec(_X.n_cols, fill::randn);

	//	set loss and y_pred
	//double loss;
	mat y_pred;

	for (long i = 0; i < max_iter; i++) {
		y_pred = sigmoid(_X * beta);
		loss = _NLL(_X, y, y_pred);
		//	stop if change is less than tol
		if (l_prev - loss < tol) return;
		l_prev = loss;
		beta = beta - lr * _NLL_grad(_X, y, y_pred);
	}
}

// supplemental function to calculate negative log likelihood under current model
double LogisticRegression::_NLL(const mat& X, const mat& y, const mat& y_pred) {
	//	for X with N rows (number of examples), and M cols (number of attributes)
	auto N = X.n_rows, M = X.n_cols;
	int order; // type of penalty depending on value of penalty
	if (penalty == "l2") order = 2;
	else order = 1;

	double norm_beta = arma::norm(beta, order);

	double nll, penalty_val;
	nll = sum(-arma::log(y_pred.elem(find(y == 1)))) - sum(log(1 - y_pred.elem(find(y == 0))));

	if (order == 2) penalty_val = (gamma / 2) * pow(norm_beta, 2);
	else penalty_val = gamma * norm_beta;

	return (nll + penalty_val) / double(N);
}


// supplemental function to calculate Gradient of the penalized negative log likelihood wrt beta
vec LogisticRegression::_NLL_grad(const mat& X, const mat& y, const mat& y_pred) {
	//	for X with N rows (number of examples), and M cols (number of attributes)
	auto N = X.n_rows, M = X.n_cols;

	// calculate delta penalty
	vec d_penalty;
	if (penalty == "l2") d_penalty = gamma * beta;
	else d_penalty = gamma * sign(beta);

	return (((y - y_pred).t() * X).t() + d_penalty) / -double(N);
}

// return predicted values with trained model
const mat LogisticRegression::predict(mat _X) {
	//	_X is a test matrix with shape of (Z, M), pred_y will be the product of _X and beta
	if (fit_intercept == true) _X = join_rows(ones(_X.n_rows), _X);
	mat y_pred = _X * beta;

	return round(sigmoid(y_pred));
}

//  supplemental functions

//The logistic sigmoid function
mat sigmoid(const mat& X) {
	return 1 / (1 + exp(-X));
};