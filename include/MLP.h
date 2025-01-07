#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "Layer.h"
#include "Activations.h"
#include "Losses.h"
#include "Optimizer.h"

class MLP {
public:
    MLP(const std::vector<int> &layers, const std::vector<ActivationType> &activations);
    ~MLP();

    Eigen::MatrixXd forward(const Eigen::MatrixXd &X);
    void backward(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, double learning_rate, const Eigen::MatrixXd &dL_dY);
    void train(const Eigen::MatrixXd &train_X, const Eigen::MatrixXd &train_Y, int epochs, double learning_rate, LossType loss_type = LossType::CROSS_ENTROPY);
    double accuracy(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
    void saveWeights(const std::string &filename);
    void loadWeights(const std::string &filename);

private:
    std::vector<Layer> network_layers;
    Optimizer* optimizer; // Pointer to the optimizer (e.g., SGD)
};

#endif
