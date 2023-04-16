/*
source file to apply LinearModels header file
*/

#include <iostream>
#include <Eigen/Dense>
#include <limits>
#include "LinearModels.hpp"
using namespace std;
using namespace Eigen;


// 1. Linear Regression
//	constructors and destructors
LinearRegression::LinearRegression() : fit_intercept(true) {};
LinearRegression::LinearRegression(const LinearRegression& source) : fit_intercept(source.fit_intercept), theta(source.theta) {};
LinearRegression::LinearRegression(bool fit_intercept1) : fit_intercept(fit_intercept1) {};
LinearRegression::~LinearRegression() {};

//	assignment operation
LinearRegression& LinearRegression::operator=(const LinearRegression& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		fit_intercept = source.fit_intercept;
		theta = source.theta;
	}
	return *this;
}

//	functions
// fit function to calculate the results and store to theta
void LinearRegression::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
	Eigen::MatrixXd X_local = X;
	//	add an addtional column if fit intercept
	if (fit_intercept == true) {
		X_local.conservativeResize(X_local.rows(), X_local.cols() + 1);
		X_local.col(X_local.cols() - 1) = Eigen::VectorXd::Ones(X_local.rows());
	}
	//	ipseduo inverse part of the equation
	Eigen::MatrixXd ipseudo_inverse = (X_local.transpose() * X_local).inverse() * X_local.transpose();

	theta = ipseudo_inverse * y;		//	save results
	return;
}

// return predicted values with trained model
Eigen::MatrixXd LinearRegression::predict(const Eigen::MatrixXd& _X) {
	Eigen::MatrixXd X_local = _X;
	//	_X is a test matrix with shape of (M, Z), pred_y will be the product of _X and theta
	if (fit_intercept == true) {
		X_local.conservativeResize(X_local.rows(), X_local.cols() + 1);
		X_local.col(X_local.cols() - 1) = Eigen::VectorXd::Ones(X_local.rows());
	}
	
	Eigen::MatrixXd y_pred = X_local * theta;
	return y_pred;
}


//	2. Ridge Regression
RidgeRegression::RidgeRegression() : fit_intercept(true), lambda(1.0) {};
RidgeRegression::RidgeRegression(const RidgeRegression& source) : fit_intercept(source.fit_intercept), theta(source.theta), lambda(source.lambda) {};
RidgeRegression::RidgeRegression(double lambda1, bool fit_intercept1) : fit_intercept(fit_intercept1), lambda(lambda1) {};
RidgeRegression::~RidgeRegression() {};

//	assignment operation
RidgeRegression& RidgeRegression::operator=(const RidgeRegression& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		fit_intercept = source.fit_intercept;
		lambda = source.lambda;
		theta = source.theta;
	}
	return *this;
}


//	functions
// fit function to calculate the results and store to theta
void RidgeRegression::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
	//	add an addtional column if fit intercept
	Eigen::MatrixXd X_local = X;
	if (fit_intercept == true) {
		X_local.conservativeResize(X_local.rows(), X_local.cols() + 1);
		X_local.col(X_local.cols() - 1) = Eigen::VectorXd::Ones(X_local.rows());
	}

	//	matrix for alpha
	Eigen::MatrixXd A = lambda * Eigen::MatrixXd::Identity(X_local.rows(), X_local.cols());
	//	ipseduo inverse part of the equation
	Eigen::MatrixXd ipseudo_inverse = (X_local.transpose() * X_local + A).inverse() * X.transpose();

	theta = ipseudo_inverse * y;		//	save results
	return;
}

// return predicted values with trained model
const Eigen::MatrixXd RidgeRegression::predict(Eigen::MatrixXd& X) {
	Eigen::MatrixXd X_local = X;
	//	_X is a test matrix with shape of (M, Z), pred_y will be the product of _X and theta
	if (fit_intercept == true) {
		X_local.conservativeResize(X_local.rows(), X_local.cols());
		X_local.col(X_local.cols() - 1) = Eigen::VectorXd::Ones(X_local.rows());
	}
	Eigen::MatrixXd y_pred = X_local * theta;

	return y_pred;
}


// 3. Logistic Regression
//	constructors and destructors
LogisticRegression::LogisticRegression() : fit_intercept(true), lambda(0.0), penalty("l2") {};
LogisticRegression::LogisticRegression(const LogisticRegression& source) : fit_intercept(source.fit_intercept), theta(source.theta), lambda(source.lambda), penalty(source.penalty) {};
LogisticRegression::LogisticRegression(double lambda1, bool fit_intercept1, string penalty1) : fit_intercept(fit_intercept1), penalty(penalty1), lambda(lambda1) {};
LogisticRegression::~LogisticRegression() {};

//	assignment operation
LogisticRegression& LogisticRegression::operator=(const LogisticRegression& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		fit_intercept = source.fit_intercept;
		lambda = source.lambda;
		penalty = source.penalty;
		theta = source.theta;
	}
	return *this;
}


//	functions
// fit function to calculate the results and store to theta
void LogisticRegression::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, double lr, double tol, long max_iter) {
	//	whether to fit an intercept
	Eigen::MatrixXd X_local = X;
	if (fit_intercept == true) {
		X_local.conservativeResize(X_local.rows(), X_local.cols() + 1);
		X_local.col(X_local.cols() - 1) = Eigen::VectorXd::Ones(X_local.rows());
	}
	//	previous loss value to store loss
	double loss, l_prev = 999999.0;
	theta = Eigen::VectorXd::Random(X_local.cols());

	//	set loss and y_pred
	//double loss;
	Eigen::MatrixXd y_pred;

	for (long i = 0; i < max_iter; ++i) {
		y_pred = sigmoid(X_local * theta);
		loss = _NLL(X_local, y, y_pred);
		//	stop if change is less than tol
		if (abs(l_prev - loss) < tol) return;
		l_prev = loss;
		theta = theta - lr * _NLL_grad(X_local, y, y_pred);
	}
}

// supplemental function to calculate negative log likelihood under current model
double LogisticRegression::_NLL(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_pred) {
	//	for X with M rows (number of examples), and N cols (number of features)
	std::size_t M = X.rows(), N = X.cols();
	// type of penalty depending on value of penalty
	double norm_theta;
	unsigned int order;
	if (penalty == "l2") {
		norm_theta = theta.lpNorm<2>();
		order = 2;
	}
	else {
		norm_theta = theta.lpNorm<1>();
		order = 1;
	};

	double nll, penalty_val;

	nll = (-y_pred.array() * (y.array() == 1).cast<double>()).log().sum();
	nll += (1 - y_pred.array() * (y.array() == 0).cast<double>()).log().sum();

	if (order == 2) penalty_val = (lambda / 2) * pow(norm_theta, 2);
	else penalty_val = lambda * norm_theta;

	return (nll + penalty_val) / double(M);
}


// supplemental function to calculate Gradient of the penalized negative log likelihood wrt theta
Eigen::VectorXd LogisticRegression::_NLL_grad(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_pred) {
	//	for X with N rows (number of examples), and M cols (number of features)
	auto M = X.rows(), N = X.cols();

	// calculate delta penalty
	Eigen::VectorXd d_penalty;
	if (penalty == "l2") d_penalty = lambda * theta;
	else d_penalty = lambda * theta.array().sign();

	return (((y - y_pred).transpose() * X).transpose() + d_penalty) / -double(M);
}

// return predicted values with trained model
const Eigen::MatrixXd LogisticRegression::predict(const Eigen::MatrixXd& X) {
	//	_X is a test matrix with shape of (M, Z), pred_y will be the product of _X and theta
	Eigen::MatrixXd X_local = X;
	if (fit_intercept == true) {
		X_local.conservativeResize(X_local.rows(), X_local.cols() + 1);
		X_local.col(X_local.cols() - 1) = Eigen::VectorXd::Ones(X_local.rows());
	};

	Eigen::MatrixXd y_pred = X_local * theta;

	return sigmoid(y_pred);
}

//  supplemental functions

//The logistic sigmoid function
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& X) {
	return 1 / (1 + -X.array().exp());
};