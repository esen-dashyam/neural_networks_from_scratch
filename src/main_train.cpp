#include <iostream>
#include "MLP.h"
#include "Utilities.h"

int main(int argc, char** argv) {
    try {
        NetworkConfig config = Utilities::parseArguments(argc, argv);

        // Convert activation strings to ActivationType enum
        std::vector<ActivationType> activations;
        for (auto &as : config.activation_strs) {
            if (as == "sigmoid") {
                activations.push_back(ActivationType::SIGMOID);
            } else if (as == "relu") {
                activations.push_back(ActivationType::RELU);
            } else if (as == "softmax") {
                activations.push_back(ActivationType::SOFTMAX);
            } else {
                throw std::runtime_error("Unknown activation function: " + as);
            }
        }

        // Initialize MLP
        MLP mlp(config.layer_sizes, activations);

        // Load training data
        Eigen::MatrixXd train_X = Utilities::loadCSV("data/train_data.csv", 784, 60000);
        Eigen::MatrixXd train_Y = Utilities::loadCSV("data/train_labels.csv", 10, 60000);

        // Train the network
        mlp.train(train_X, train_Y, config.epochs, config.learning_rate, LossType::CROSS_ENTROPY);

        // Save the trained weights the main problem I had was that the weights were not being saved and I had to do everything over and over again
        mlp.saveWeights("weights.bin");
        std::cout << "Weights saved to weights.bin\n";

        // to check accuracy on a subset of training data
        Eigen::MatrixXd subset_X = train_X.leftCols(1000);
        Eigen::MatrixXd subset_Y = train_Y.leftCols(1000);
        double acc = mlp.accuracy(subset_X, subset_Y);
        std::cout << "Accuracy on 1000-sample subset: " << acc * 100 << "%\n";

    } catch (const std::runtime_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
