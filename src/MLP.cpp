// MLP.cpp
#include "MLP.h"
#include "Losses.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream> // Added to resolve std::ofstream and std::ifstream errors

// Helper function to shuffle data
static void shuffleData(Eigen::MatrixXd &X, Eigen::MatrixXd &Y) {
    std::vector<int> indices(X.cols());
    for (int i = 0; i < X.cols(); i++) indices[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::MatrixXd X_shuffled(X.rows(), X.cols());
    Eigen::MatrixXd Y_shuffled(Y.rows(), Y.cols());
    for (int i = 0; i < (int)indices.size(); i++) {
        X_shuffled.col(i) = X.col(indices[i]);
        Y_shuffled.col(i) = Y.col(indices[i]);
    }
    X = X_shuffled;
    Y = Y_shuffled;
}

MLP::MLP(const std::vector<int> &layers, const std::vector<ActivationType> &activations) {
    if (activations.size() != layers.size() - 1) {
        throw std::runtime_error("Number of activations must be one less than number of layer sizes.");
    }

    for (size_t i = 0; i < activations.size(); ++i) {
        Layer layer(layers[i], layers[i+1], activations[i]);
        network_layers.push_back(layer);
    }

    optimizer = new SGD(0.01);
}

MLP::~MLP() {
    delete optimizer;
}

Eigen::MatrixXd MLP::forward(const Eigen::MatrixXd &X) {
    Eigen::MatrixXd out = X;
    for (auto &layer : network_layers) {
        out = layer.forward(out);
    }
    return out;
}

void MLP::backward(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, double learning_rate, const Eigen::MatrixXd &dL_dY) {
    delete optimizer;
    optimizer = new SGD(learning_rate);

    Eigen::MatrixXd grad = dL_dY;
    for (int i = (int)network_layers.size() - 1; i >= 0; --i) {
        Layer &layer = network_layers[i];
        Eigen::MatrixXd dZ = grad.array() * Activations::derivative(layer.output_cache, layer.activation_type).array();

        Eigen::MatrixXd dW = dZ * layer.input_cache.transpose();
        Eigen::VectorXd db = dZ.rowwise().sum();

        optimizer->updateWeights(layer.W, layer.b, dW, db);

        grad = layer.W.transpose() * dZ;
    }
}

void MLP::train(const Eigen::MatrixXd &train_X, const Eigen::MatrixXd &train_Y, int epochs, double learning_rate, LossType loss_type) {
    if (train_X.cols() == 0 || train_Y.cols() == 0) {
        throw std::runtime_error("Empty training data provided.");
    }

    Eigen::MatrixXd X = train_X;
    Eigen::MatrixXd Y = train_Y;

    int batch_size = 64;
    int num_samples = X.cols();
    if (num_samples < batch_size) {
        throw std::runtime_error("Not enough samples to form a single batch.");
    }
    int num_batches = num_samples / batch_size;

    for (int e = 0; e < epochs; ++e) {
        shuffleData(X, Y);
        double epoch_loss = 0.0;
        for (int b = 0; b < num_batches; b++) {
            int start = b * batch_size;
            Eigen::MatrixXd X_batch = X.block(0, start, X.rows(), batch_size);
            Eigen::MatrixXd Y_batch = Y.block(0, start, Y.rows(), batch_size);

            Eigen::MatrixXd Y_pred = forward(X_batch);

            double loss;
            Eigen::MatrixXd dL_dY;
            if (loss_type == LossType::CROSS_ENTROPY) {
                loss = Losses::crossEntropy(Y_pred, Y_batch);
                dL_dY = Losses::crossEntropy_derivative(Y_pred, Y_batch);
            } else {
                loss = Losses::MSE(Y_pred, Y_batch);
                dL_dY = Losses::MSE_derivative(Y_pred, Y_batch);
            }

            epoch_loss += loss;
            backward(X_batch, Y_batch, learning_rate, dL_dY);
        }

        epoch_loss /= num_batches;
        if (e % 100 == 0 || e == epochs - 1) {
            std::cout << "Epoch " << e << ", Loss: " << epoch_loss << std::endl;
        }
    }
}

double MLP::accuracy(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y) {
    if (X.cols() == 0 || Y.cols() == 0) {
        throw std::runtime_error("Empty data provided for accuracy calculation.");
    }

    Eigen::MatrixXd Y_pred = forward(X);
    int correct = 0;
    for (int i = 0; i < Y_pred.cols(); i++) {
        Eigen::Index predClass, trueClass;
        Y_pred.col(i).maxCoeff(&predClass);
        Y.col(i).maxCoeff(&trueClass);
        if (predClass == trueClass) correct++;
    }
    return static_cast<double>(correct) / Y_pred.cols();
}

void MLP::saveWeights(const std::string &filename) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Error opening file for saving weights: " + filename);
    }

    int num_layers = (int)network_layers.size();
    f.write((char*)&num_layers, sizeof(num_layers));
    for (auto &layer : network_layers) {
        int rows = (int)layer.W.rows();
        int cols = (int)layer.W.cols();
        f.write((char*)&rows, sizeof(rows));
        f.write((char*)&cols, sizeof(cols));
        f.write((char*)layer.W.data(), rows * cols * sizeof(double));

        int b_size = (int)layer.b.size();
        f.write((char*)&b_size, sizeof(b_size));
        f.write((char*)layer.b.data(), b_size * sizeof(double));

        int act = static_cast<int>(layer.activation_type);
        f.write((char*)&act, sizeof(act));
    }
    f.close();
}

void MLP::loadWeights(const std::string &filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Error opening file for loading weights: " + filename);
    }

    int num_layers;
    f.read((char*)&num_layers, sizeof(num_layers));
    if (num_layers <= 0) {
        throw std::runtime_error("Invalid number of layers in weights file.");
    }

    network_layers.clear();
    network_layers.reserve(num_layers);
    for (int i = 0; i < num_layers; i++) {
        int rows, cols;
        f.read((char*)&rows, sizeof(rows));
        f.read((char*)&cols, sizeof(cols));
        if (rows <= 0 || cols <= 0) {
            throw std::runtime_error("Invalid layer dimensions in weights file.");
        }
        Eigen::MatrixXd W(rows, cols);
        f.read((char*)W.data(), rows * cols * sizeof(double));

        int b_size;
        f.read((char*)&b_size, sizeof(b_size));
        if (b_size <= 0) {
            throw std::runtime_error("Invalid bias size in weights file.");
        }
        Eigen::VectorXd b(b_size);
        f.read((char*)b.data(), b_size * sizeof(double));

        int act_int;
        f.read((char*)&act_int, sizeof(act_int));
        ActivationType act = static_cast<ActivationType>(act_int);

        Layer layer(cols, rows, act); // Note: input_size = cols, output_size = rows
        layer.W = W;
        layer.b = b;
        layer.activation_type = act;
        network_layers.push_back(layer);
    }
    f.close();
}
