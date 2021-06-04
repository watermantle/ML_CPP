/*
Header files to implement linear models
X: mat type with shape (N, M), a dataset consisting of 'N' examples each of dimension of 'M'
y: mat type with shape (N, K), the targets for each of the 'N' examples in 'X', where each target has dimension `K` 

1. Linear Regression
To apply Ordinary Least Squares with normal equation with formula:
\hat{\beta} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}

2. Ridge Regression
To apply Ridge regression with normal equation:
 \hat{\beta} \left(\mathbf{X}^\top \mathbf{X} +
 \alpha \mathbf{I} \right)^{-1}\mathbf{X}^\top \mathbf{y}
 
where alpha is the paramater for L2 regulization, a greater value has a larger penalty

3. Logistic Regression
To minimize the following loss function:

- \log \mathcal{L}(\mathbf{b}, \mathbf{y}) = -\frac{1}{N} \left[
        \left(
            \sum_{i=0}^N y_i \log(\hat{y}_i) +
                (1-y_i) \log(1-\hat{y}_i)
        \right) - R(\mathbf{b}, \gamma)
    \right]

Where:
R(\mathbf{b}, \gamma) = \left\{
                \begin{array}{lr}
                    \frac{\gamma}{2} ||\mathbf{beta}||_2^2 & :\texttt{ penalty = 'l2'}\\
                    \gamma ||\beta||_1 & :\texttt{ penalty = 'l1'}
                \end{array}
                \right
is a regularization penalty, '\gamma' is a regularization weight, 'N' is the number of examples in y
fit with gradient descent algo

*/
#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;


#ifndef LinearModels_HPP
#define LinearModels_HPP

class LinearRegression {
private:
    //  Whether to fit intercept, with default value true
    bool fit_intercept;
    
public:
    //  a vector to store coefficents of the results with defult shape of 100 X 1
    vec beta; 

    //  constructors
    LinearRegression();
    LinearRegression(bool fit_intercept1);
    LinearRegression(const LinearRegression& source);
    ~LinearRegression();

    //  Assignment operator
    LinearRegression& operator = (const LinearRegression& source);

    //  functions
    const void fit(mat X, const mat& y);   // fit function to calculate the results and store to beta
    const mat predict(mat X);   // return predicted values with trained model
};


class RidgeRegression {
private:
    //  Whether to fit intercept, with default value true
    bool fit_intercept;
    //  Regulation paramater 
    double alpha;

public:
    //  a vector to store coefficents of the results with defult shape of 100 X 1
    vec beta;

    //  constructors
    RidgeRegression();
    RidgeRegression(double alpha1, bool fit_intercept1=true);
    RidgeRegression(const RidgeRegression& source);
    ~RidgeRegression();

    //  Assignment operator
    RidgeRegression& operator = (const RidgeRegression& source);

    //  functions
    const void fit(mat X, const mat& y);    // fit function to calculate the results and store to beta
    const mat predict(mat X);   // return predicted values with trained model
};


//  Logistic regression with gradient descent optimizer 
class LogisticRegression {
private:
    //  regularization paramater
    double gamma;
    //  regularization type with l2 as default/
    string penalty;
    //  Whether to fit intercept, with default value true
    bool fit_intercept;

public:
    //  a vector to store coefficents of the results with defult shape of 100 X 1
    vec beta;
    //  constructors
    LogisticRegression();
    LogisticRegression(double gamma1 = 0, bool fit_intercept1 = true, string penalty = "l2");
    LogisticRegression(const LogisticRegression& source);
    ~LogisticRegression();

    //  Assignment operator
    LogisticRegression& operator = (const LogisticRegression& source);

    //  functions
    // fit function to calculate the results and store to beta. Apply gradient descent algo to minimize loss function
    const void fit(mat X, const mat& y, double lr = 0.01, double tol = 1e-7, long max_iter = 1e7);
    const mat predict(mat X);   // return predicted values with trained model
    double _NLL(const mat& X, const mat& y, const mat& y_pred); // supplemental function to calculate negative log likelihood under current model
    vec _NLL_grad(const mat& X, const mat& y, const mat& y_pred); // supplemental function to calculate Gradient of the penalized negative log likelihood wrt beta
};

//  supplemental functions

//The logistic sigmoid function
mat sigmoid(const mat& X);
#endif;

