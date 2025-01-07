#include <iostream>
#include "MLP.h"
#include "Utilities.h"

int main(int argc, char** argv) {
    try {
        // Define the same architecture as training
        std::vector<int> layer_sizes = {784, 256, 128, 128, 128, 10};
        std::vector<ActivationType> activations = {ActivationType::SIGMOID, ActivationType::RELU, ActivationType::SIGMOID, ActivationType::RELU, ActivationType::SOFTMAX};

        MLP mlp(layer_sizes, activations);
        mlp.loadWeights("weights.bin"); // Load trained weights

        // Load single image
        Eigen::MatrixXd single_image = Utilities::loadCSV("data2/single_image_label_0_9.csv", 784, 1);
        Eigen::MatrixXd output = mlp.forward(single_image);

        std::cout << "Predicted probabilities:\n" << output << std::endl;

        // Calculate and print the sum of probabilities to verify softmax
        double sum = output.col(0).sum();
        std::cout << "Sum of probabilities: " << sum << std::endl;

        Eigen::Index maxIndex;
        output.col(0).maxCoeff(&maxIndex);
        std::cout << "Predicted class: " << maxIndex << std::endl;

    } catch (const std::runtime_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
