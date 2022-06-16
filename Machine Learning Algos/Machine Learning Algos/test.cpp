#include <iostream>
#include <armadillo>
#include <random>
#include <math.h>
#include "LinearModels/LinearModels.hpp"
#include "datasets/datasets_generator.hpp"


using namespace std;
using namespace arma;

int main() {
    tuple<mat, vec> data = make_regression(10, 1, 100);
    mat X = get<0>(data);
    vec y = get<1>(data);

    LinearRegression reg = LinearRegression();
    reg.fit(X, y);
    vec y_pred = reg.predict(X);
    
    double cost = sum(pow((y - y_pred), 2));
    cout << "the cost is " << cost << endl;
    cout << "the coeff are" << reg.theta << endl;

    return 0;
}

