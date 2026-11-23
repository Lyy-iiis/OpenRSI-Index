#include "flow.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

struct Score {
    double sum = 0;
    int count = 0;
    void add(double error) { sum += error; ++count; }
};

struct Pair {
    cv::Mat first, second, truth, invalid, occluded, boundary;
};

cv::Mat image(const fs::path& file) {
    auto result = cv::imread(file.string(), cv::IMREAD_GRAYSCALE);
    if (result.empty()) throw std::runtime_error("missing image " + file.string());
    return result;
}

fs::path frame(const fs::path& root, const std::string& kind, const std::string& scene,
               int index, const std::string& extension) {
    std::ostringstream name;
    name << "frame_" << std::setw(4) << std::setfill('0') << index << extension;
    return root / "training" / kind / scene / name.str();
}

Pair loadPair(const fs::path& root, const std::string& scene, int index) {
    Pair p;
    p.first = image(frame(root, "final", scene, index, ".png"));
    p.second = image(frame(root, "final", scene, index + 1, ".png"));
    p.invalid = image(frame(root, "invalid", scene, index, ".png"));
    p.occluded = image(frame(root, "occlusions", scene, index, ".png"));
    p.truth = cv::readOpticalFlow(frame(root, "flow", scene, index, ".flo").string());
    for (const auto& m : {p.second, p.truth, p.invalid, p.occluded})
        if (m.size() != p.first.size()) throw std::runtime_error("missing or mismatched annotation");
    cv::Mat edge, distance;
    cv::morphologyEx(p.occluded, edge, cv::MORPH_GRADIENT, cv::Mat());
    cv::distanceTransform(edge == 0, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    p.boundary = distance < 10;
    return p;
}

std::array<Score, 3> evaluate(const Pair& p, const cv::Mat& flow) {
    if (flow.size() != p.first.size() || flow.type() != CV_32FC2 || !cv::checkRange(flow))
        throw std::runtime_error("invalid predicted flow");
    std::array<Score, 3> scores{};
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            if (p.invalid.at<uchar>(y, x)) continue;
            const auto gt = p.truth.at<cv::Vec2f>(y, x);
            if (!std::isfinite(gt[0]) || !std::isfinite(gt[1]) ||
                std::max(std::abs(gt[0]), std::abs(gt[1])) >= 1e9f)
                throw std::runtime_error("invalid ground truth outside invalid mask");
            double error = cv::norm(flow.at<cv::Vec2f>(y, x) - gt);
            scores[0].add(error);
            if (p.occluded.at<uchar>(y, x)) scores[1].add(error);
            if (p.boundary.at<uchar>(y, x)) scores[2].add(error);
        }
    }
    return scores;
}

int main(int argc, char** argv) try {
    if (argc != 5) throw std::invalid_argument("usage: flow_bench DATASET_ROOT PAIRS_TSV dev|holdout ODD_REPEATS");
    const std::string split = argv[3];
    const int repeats = std::stoi(argv[4]);
    if ((split != "dev" && split != "holdout") || repeats < 3 || repeats % 2 == 0)
        throw std::invalid_argument("expected dev or holdout, and an odd repeat count >= 3");
    std::ifstream manifest(argv[2]);
    if (!manifest) throw std::runtime_error("cannot open pair manifest");
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
    const std::array<std::string, 5> names{"fast", "medium", "extra-iteration", "full-grid", "adaptive"};
    std::vector<FlowEstimator> methods;
    for (const auto& name : names) methods.emplace_back(name);
    std::cout << "split,scene,frame,method,median_ms,epe_sum,n,occ_sum,occ_n,boundary_sum,boundary_n\n";
    int pairIndex = 0;
    std::string group, scene;
    int index;
    while (manifest >> group >> scene >> index) {
        if (group != split) continue;
        const auto p = loadPair(argv[1], scene, index);
        std::array<std::vector<double>, names.size()> times;
        std::array<cv::Mat, names.size()> outputs;
        // Rotate methods both between repetitions and between pairs.
        for (int rep = -1; rep < repeats; ++rep) {
            for (size_t j = 0; j < names.size(); ++j) {
                const size_t m = (j + pairIndex + rep + 1) % names.size();
                const auto start = std::chrono::steady_clock::now();
                auto prediction = methods[m].run(p.first, p.second);
                const auto elapsed = std::chrono::steady_clock::now() - start;
                if (rep >= 0) times[m].push_back(std::chrono::duration<double,std::milli>(elapsed).count());
                if (rep == repeats - 1) outputs[m] = std::move(prediction);
            }
        }
        for (size_t m = 0; m < names.size(); ++m) {
            std::sort(times[m].begin(), times[m].end());
            std::cout << std::setprecision(12) << split << ',' << scene << ',' << index
                      << ',' << names[m] << ',' << times[m][repeats / 2];
            for (const auto& score : evaluate(p, outputs[m]))
                std::cout << ',' << score.sum << ',' << score.count;
            std::cout << '\n';
        }
        ++pairIndex;
    }
    if (!manifest.eof() || pairIndex == 0) throw std::runtime_error("invalid or empty manifest");
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
