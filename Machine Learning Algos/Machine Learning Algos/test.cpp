#include <iostream>
#include <armadillo>
#include <random>
#include <math.h>
#include "LinearModels/LinearModels.hpp"
#include "datasets/datasets_generator.hpp"


using namespace std;
using namespace arma;


int main() {
    // test regression functions
    
   /* tuple<mat, vec> data = make_classification(1000, 6, 2);
    mat X = get<0>(data);
    vec y = get<1>(data);*/
    
    

    tuple<mat, vec> iris = dataloader("iris");
    mat X = get<0>(iris);
    vec y = get<1>(iris);
    
    uvec idx = find(y < 2);
    mat X_train = X.rows(idx);
    vec y_train = y.rows(idx);

    LogisticRegression log_reg = LogisticRegression(true);
    log_reg.fit(X_train, y_train);

    vec y_pred = log_reg.predict(X_train);
    cout << sum(abs(y_pred - y_train)) << endl;
    return 0;
}

