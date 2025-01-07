#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>

class Optimizer {
public:
    virtual ~Optimizer() {}
    virtual void updateWeights(Eigen::MatrixXd &W, Eigen::VectorXd &b,
                               const Eigen::MatrixXd &dW, const Eigen::VectorXd &db) = 0;
};

class SGD : public Optimizer {
public:
    SGD(double lr) : learning_rate(lr) {}
    virtual void updateWeights(Eigen::MatrixXd &W, Eigen::VectorXd &b,
                               const Eigen::MatrixXd &dW, const Eigen::VectorXd &db) override {
        W -= learning_rate * dW;
        b -= learning_rate * db;
    }

private:
    double learning_rate;
};

#endif
