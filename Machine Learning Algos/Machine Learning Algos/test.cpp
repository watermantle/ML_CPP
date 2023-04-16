#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "LinearModels/LinearModels.hpp"
#include "datasets/datasets_generator.hpp"


using namespace std;
using namespace Eigen;


int main() {
    pair<Eigen::MatrixXd, Eigen::VectorXd> iris = dataloader("iris");
    Eigen::MatrixXd x = iris.first;
    Eigen::VectorXd y = iris.second;

    vector<int> idx;
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) < 2) idx.push_back(i);
    }

    Eigen::MatrixXd x_train(idx.size(), x.cols());
    Eigen::VectorXd y_train(idx.size());

    for (int i = 0; i < idx.size(); ++i) {
        x_train.row(i) = x.row(idx[i]);
        y_train(i) = y(idx[i]);
    }

     //modify this line: use the existing constructor with two parameters
    LogisticRegression log_reg = LogisticRegression(true);
    log_reg.fit(x_train, y_train, 0.01, 0.001, 100);

    Eigen::VectorXd y_pred = log_reg.predict(x_train);

    cout << (y_pred - y_train).cwiseAbs().sum() << endl;

    return 0;
}

