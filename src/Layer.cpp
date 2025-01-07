#include "Layer.h"
#include "Activations.h"
#include <random>
#include <cmath>
#include <stdexcept>

Layer::Layer(int input_size, int output_size, ActivationType activation) : W(output_size, input_size), b(output_size) {
    activation_type = activation;
    // Initialize weights with small random values using Xavier/Glorot initialization
    double limit;
    if (activation_type == ActivationType::SIGMOID || activation_type == ActivationType::SOFTMAX) {
        limit = std::sqrt(6.0 / (input_size + output_size));
    } else { // ReLU
        limit = std::sqrt(2.0 / input_size);
    }
    // Initialize W with uniform distribution [-limit, limit]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-limit, limit);
    for(int i = 0; i < W.size(); ++i) {
        W.data()[i] = dis(gen);
    }
    // Initialize biases to zero
    b.setZero();
}

Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd &input) {
    input_cache = input; // Cache for backpropagation
    Eigen::MatrixXd Z = (W * input).colwise() + b;
    output_cache = Activations::activate(Z, activation_type);
    return output_cache;
}
