#include "Utilities.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdexcept>

static std::vector<std::string> splitString(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

NetworkConfig Utilities::parseArguments(int argc, char** argv) {
    NetworkConfig config;
    // Defaults
    config.layer_sizes = {784, 128, 10};
    config.activation_strs = {"sigmoid", "sigmoid"};
    config.epochs = 10;
    config.learning_rate = 0.01;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--sizes" || arg == "-s") && i + 1 < argc) {
            config.layer_sizes = parseLayerSizes(argv[++i]);
        } else if ((arg == "--activations" || arg == "-a") && i + 1 < argc) {
            config.activation_strs = parseActivations(argv[++i]);
        } else if ((arg == "--epochs" || arg == "-e") && i + 1 < argc) {
            std::string val_str = argv[++i];
            try {
                config.epochs = std::stoi(val_str);
                if (config.epochs <= 0) {
                    throw std::runtime_error("Number of epochs must be positive.");
                }
            } catch (...) {
                throw std::runtime_error("Invalid value for --epochs. Must be an integer.");
            }
        } else if ((arg == "--lr" || arg == "--learning_rate") && i + 1 < argc) {
            std::string val_str = argv[++i];
            try {
                config.learning_rate = std::stod(val_str);
                if (config.learning_rate <= 0.0) {
                    throw std::runtime_error("Learning rate must be positive.");
                }
            } catch (...) {
                throw std::runtime_error("Invalid value for --lr. Must be a floating point number.");
            }
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    // Validate activation count
    if (config.activation_strs.size() != config.layer_sizes.size() - 1) {
        throw std::runtime_error("Number of activations must be one less than number of layer sizes.");
    }

    return config;
}

std::vector<int> Utilities::parseLayerSizes(const std::string &sizes_str) {
    auto parts = splitString(sizes_str, ',');
    std::vector<int> sizes;
    for (auto &p : parts) {
        try {
            int size = std::stoi(p);
            if (size <= 0) {
                throw std::runtime_error("Layer sizes must be positive integers.");
            }
            sizes.push_back(size);
        } catch (...) {
            throw std::runtime_error("Invalid layer size value: " + p);
        }
    }
    return sizes;
}

std::vector<std::string> Utilities::parseActivations(const std::string &act_str) {
    auto parts = splitString(act_str, ',');
    std::vector<std::string> activations;
    for (auto &p : parts) {
        if (p.empty()) {
            throw std::runtime_error("Activation function names cannot be empty.");
        }
        activations.push_back(p);
    }
    return activations;
}

Eigen::MatrixXd Utilities::loadCSV(const std::string &filename, int rows, int cols) {
    Eigen::MatrixXd mat(rows, cols);
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    int row = 0;
    while (std::getline(file, line) && row < rows) {
        std::stringstream ss(line);
        std::string val;
        int col = 0;
        while (std::getline(ss, val, ',') && col < cols) {
            try {
                mat(row, col) = std::stod(val);
            } catch (...) {
                throw std::runtime_error("Non-numeric value found in CSV file: " + filename + " at row " + std::to_string(row) + ", column " + std::to_string(col));
            }
            col++;
        }
        if (col < cols) {
            throw std::runtime_error("Not enough columns in " + filename + " at row " + std::to_string(row));
        }
        row++;
    }
    if (row < rows) {
        throw std::runtime_error("Not enough rows in " + filename + ". Expected " + std::to_string(rows) + ", got " + std::to_string(row));
    }

    file.close();
    return mat;
}
