#ifndef LOSSES_H
#define LOSSES_H

#include <Eigen/Dense>

enum class LossType {
    MSE,
    CROSS_ENTROPY
};

class Losses {
public:
    static double MSE(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true);
    static Eigen::MatrixXd MSE_derivative(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true);

    static double crossEntropy(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true);
    static Eigen::MatrixXd crossEntropy_derivative(const Eigen::MatrixXd &Y_pred, const Eigen::MatrixXd &Y_true);
};

#endif
