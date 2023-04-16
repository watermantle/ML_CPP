/*
Header files to implement linear models
X: mat type with shape (M, N), a dataset consisting of 'M' examples each of dimension of 'N'
y: mat type with shape (M, K), the targets for each of the 'M' examples in 'X', where each target has dimension `K` 

1. Linear Regression
To apply Ordinary Least Squares with normal equation with formula:
\mathbf{theta}= \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}

2. Ridge Regression
To apply Ridge regression with normal equation:
 \mathbf{theta} \left(\mathbf{X}^\top \mathbf{X} +
 \lambda \mathbf{I} \right)^{-1}\mathbf{X}^\top \mathbf{y}
 
where alpha is the paramater for L2 regulization, a greater value has a larger penalty

3. Logistic Regression
To minimize the following loss function:

- \log (\mathcal{L}(\mathbf{theta})) = -\frac{1}{M} \left[
        \left(
            \sum_{i=0}^M \y^((i)) \log(\h_theta(x^((i)))) +
                (1-y^((i))) \log(1-\h_theta(x^((i))))
        \right) - R(\mathbf{theta}, \lambda)
    \right]

Where:
R(mathbf{theta}, lambda) = {(lambda/2||\mathbf{theta}||_2^2\ :\ "penalty" = 'l2'),
(lambda||\mathbf{theta}||_1:\ "penalty" = 'l1'):}

is a regularization penalty, '\lambda' is a regularization weight, 'M' is the number of examples in y
fit with gradient descent algo

*/
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;


#ifndef LinearModels_HPP
#define LinearModels_HPP

class LinearRegression {
private:
    //  Whether to fit intercept, with default value true
    bool fit_intercept;
    
public:
    //  a vector to store coefficents of the results with defult shape of 100 X 1
    Eigen::VectorXd theta;

    //  constructors
    LinearRegression();
    LinearRegression(bool fit_intercept1);
    LinearRegression(const LinearRegression& source);
    ~LinearRegression();

    //  Assignment operator
    LinearRegression& operator = (const LinearRegression& source);

    //  functions
    void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y);   // fit function to calculate the results and store to theta
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X);   // return predicted values with trained model
};


class RidgeRegression {
private:
    //  Whether to fit intercept, with default value true
    bool fit_intercept;
    //  Regulation paramater 
    double lambda;

public:
    //  a vector to store coefficents of the results with defult shape of 100 X 1
    Eigen::VectorXd theta;

    //  constructors
    RidgeRegression();
    RidgeRegression(double lambda1, bool fit_intercept1=true);
    RidgeRegression(const RidgeRegression& source);
    ~RidgeRegression();

    //  Assignment operator
    RidgeRegression& operator = (const RidgeRegression& source);

    //  functions
    void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y);    // fit function to calculate the results and store to theta
    const Eigen::MatrixXd predict(Eigen::MatrixXd& X);   // return predicted values with trained model
};


//  Logistic regression with gradient descent optimizer 
class LogisticRegression {
private:
    //  regularization paramater
    double lambda;
    //  regularization type with l2 as default/
    string penalty;
    //  Whether to fit intercept, with default value true
    bool fit_intercept;
    double _NLL(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_pred); // supplemental function to calculate negative log likelihood under current model
    Eigen::VectorXd _NLL_grad(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_pred); // supplemental function to calculate Gradient of the penalized negative log likelihood wrt theta

public:
    //  a vector to store coefficents of the results with defult shape of 100 X 1
    Eigen::VectorXd theta;
    //  constructors
    LogisticRegression();
    LogisticRegression(double lambda1 = 0, bool fit_intercept1 = true, string penalty = "l2");
    LogisticRegression(const LogisticRegression& source);
    ~LogisticRegression();

    //  Assignment operator
    LogisticRegression& operator = (const LogisticRegression& source);

    //  functions
    // fit function to calculate the results and store to theta. Apply gradient descent algo to minimize loss function
    void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, double lr = 0.01, double tol = 1e-7, long max_iter = 1e7);
    const Eigen::MatrixXd predict(const Eigen::MatrixXd& X);   // return predicted values with trained model   
};

//  supplemental functions

//The logistic sigmoid function
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& X);
#endif;