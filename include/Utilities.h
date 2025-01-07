#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <string>
#include <Eigen/Dense>

struct NetworkConfig {
    std::vector<int> layer_sizes;
    std::vector<std::string> activation_strs;
    int epochs;
    double learning_rate;
};

class Utilities {
public:
    static NetworkConfig parseArguments(int argc, char** argv);
    static std::vector<int> parseLayerSizes(const std::string &sizes_str);
    static std::vector<std::string> parseActivations(const std::string &act_str);
    static Eigen::MatrixXd loadCSV(const std::string &filename, int rows, int cols);
};

#endif
