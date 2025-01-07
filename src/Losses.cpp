#include "Losses.h"
#include <cmath>
#include <stdexcept>

// Mean Squared Error
double Losses::MSE(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true) {
    Eigen::MatrixXd diff = Y_pred - Y_true;
    return diff.array().square().mean();
}

Eigen::MatrixXd Losses::MSE_derivative(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true) {
    return 2.0 * (Y_pred - Y_true) / Y_pred.cols();
}

// Cross-Entropy Loss
double Losses::crossEntropy(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true) {
    // Add epsilon to avoid log(0)
    double epsilon = 1e-12;
    Eigen::MatrixXd clipped = Y_pred.array().max(epsilon).min(1.0 - epsilon);
    Eigen::MatrixXd log_preds = clipped.array().log();
    double loss = -(Y_true.array() * log_preds.array()).sum() / Y_pred.cols();
    return loss;
}

// Derivative of Cross-Entropy Loss w.r.t. Y_pred
Eigen::MatrixXd Losses::crossEntropy_derivative(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true) {
    // When using softmax activation with cross-entropy loss,
    // the derivative simplifies to (Y_pred - Y_true) / N
    return (Y_pred - Y_true) / Y_pred.cols();
}
