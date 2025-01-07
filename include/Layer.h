#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include "Activations.h"

class Layer {
public:
    Layer(int input_size, int output_size, ActivationType activation);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

    Eigen::MatrixXd W; // Weights matrix (output_size x input_size)
    Eigen::VectorXd b; // Bias vector (output_size)

    Eigen::MatrixXd input_cache;  // Cache of inputs for backpropagation
    Eigen::MatrixXd output_cache; // Cache of outputs for backpropagation

    ActivationType activation_type;
};

#endif
