// Activations.cpp
#include "Activations.h"
#include <cmath>
#include <stdexcept>

// Sigmoid activation function
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &Z) {
    return 1.0 / (1.0 + (-Z.array()).exp());
}

// Derivative of sigmoid
Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd &A) {
    return A.array() * (1.0 - A.array());
}

// ReLU activation function
Eigen::MatrixXd relu(const Eigen::MatrixXd &Z) {
    return Z.array().max(0.0);
}

// Derivative of ReLU
Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd &A) {
    return (A.array() > 0).cast<double>();
}

// Softmax activation function with numerical stability
Eigen::MatrixXd softmax(const Eigen::MatrixXd &Z) {
    // Compute the maximum coefficient for each column (row-wise max)
    Eigen::RowVectorXd max_coeff = Z.colwise().maxCoeff();
    
    // Subtract the max coefficient from each row to ensure numerical stability
    Eigen::MatrixXd shifted = Z.rowwise() - max_coeff;
    
    // Exponentiate the shifted values
    Eigen::MatrixXd exps = shifted.array().exp();
    
    // Compute the sum of exponentials for each column
    Eigen::RowVectorXd sums = exps.colwise().sum();
    
    // Divide exponentials by the sum to get probabilities
    return exps.array().rowwise() / sums.array();
}

// Derivative of Softmax is handled with Cross-Entropy loss
Eigen::MatrixXd softmax_derivative(const Eigen::MatrixXd &A) {
    // Not used directly because derivative is handled with cross-entropy
    return Eigen::MatrixXd::Ones(A.rows(), A.cols()); // Placeholder
}


Eigen::MatrixXd Activations::activate(const Eigen::MatrixXd &Z, ActivationType type) {
    switch(type) {
        case ActivationType::SIGMOID:
            return sigmoid(Z);
        case ActivationType::RELU:
            return relu(Z);
        case ActivationType::SOFTMAX:
            return softmax(Z);
        default:
            throw std::runtime_error("Unknown activation type.");
    }
}

Eigen::MatrixXd Activations::derivative(const Eigen::MatrixXd &A, ActivationType type) {
    switch(type) {
        case ActivationType::SIGMOID:
            return sigmoid_derivative(A);
        case ActivationType::RELU:
            return relu_derivative(A);
        case ActivationType::SOFTMAX:
            return Eigen::MatrixXd::Ones(A.rows(), A.cols());
        default:
            throw std::runtime_error("Unknown activation type.");
    }
}
