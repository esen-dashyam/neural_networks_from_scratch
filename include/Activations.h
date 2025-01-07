#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <Eigen/Dense>
#include <vector>
#include <string>

enum class ActivationType {
    SIGMOID,
    RELU,
    SOFTMAX
};

class Activations {
public:
    static Eigen::MatrixXd activate(const Eigen::MatrixXd &Z, ActivationType type);
    static Eigen::MatrixXd derivative(const Eigen::MatrixXd &A, ActivationType type);
};

#endif
