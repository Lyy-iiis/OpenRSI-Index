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
#include <random>
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

Pair loadMiddlebury(const fs::path& root, const std::string& scene, int index) {
    Pair p;
    const auto dir = root / "other-data" / scene;
    p.first = image(dir / ("frame" + std::to_string(index) + ".png"));
    p.second = image(dir / ("frame" + std::to_string(index + 1) + ".png"));
    p.truth = cv::readOpticalFlow((root / "other-gt-flow" / scene /
                                  ("flow" + std::to_string(index) + ".flo")).string());
    if (p.truth.size() != p.first.size() || p.second.size() != p.first.size())
        throw std::runtime_error("missing or mismatched Middlebury input: " + scene);
    // Unknown Middlebury vectors are written with a large sentinel in either component.
    cv::Mat valid;
    cv::inRange(p.truth, cv::Scalar(-1e9, -1e9), cv::Scalar(1e9, 1e9), valid);
    cv::bitwise_not(valid, p.invalid);
    return p;
}

// Fisher-Yates with a bounded, unbiased draw and reproducible engine output.
std::vector<uint8_t> randomSelection(RefinementSites sites, uint32_t seed) {
    std::vector<uint8_t> mask(sites.eligible, 0);
    std::fill_n(mask.begin(), sites.checked, 1);
    std::mt19937 engine(seed);
    for (uint32_t size = static_cast<uint32_t>(mask.size()); size > 1; --size) {
        const uint32_t threshold = (uint32_t(0) - size) % size;
        uint32_t value;
        do { value = engine(); } while (value < threshold);
        std::swap(mask[size - 1], mask[value % size]);
    }
    return mask;
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
                std::max(std::abs(gt[0]), std::abs(gt[1])) > 1e9f)
                throw std::runtime_error("invalid ground truth outside invalid mask");
            double error = cv::norm(flow.at<cv::Vec2f>(y, x) - gt);
            scores[0].add(error);
            if (!p.occluded.empty() && p.occluded.at<uchar>(y, x)) scores[1].add(error);
            if (!p.boundary.empty() && p.boundary.at<uchar>(y, x)) scores[2].add(error);
        }
    }
    return scores;
}

int main(int argc, char** argv) try {
    if (argc != 5 && argc != 6)
        throw std::invalid_argument("usage: flow_bench DATASET_ROOT PAIRS_TSV SPLIT ODD_REPEATS [allocation]");
    const bool allocation = argc == 6;
    if (allocation && std::string(argv[5]) != "allocation")
        throw std::invalid_argument("expected allocation mode");
    const std::string split = argv[3];
    const int repeats = std::stoi(argv[4]);
    if ((split != "dev" && split != "holdout" && !(allocation && split == "middlebury")) || repeats < 3 || repeats % 2 == 0)
        throw std::invalid_argument("invalid split or repeat count (expected an odd number >= 3)");
    std::ifstream manifest(argv[2]);
    if (!manifest) throw std::runtime_error("cannot open pair manifest");
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
    constexpr std::array<uint32_t, 3> seeds{17, 29, 43};
    std::vector<std::string> names{"fast", "medium", "extra-iteration", "full-grid", "adaptive"};
    if (allocation) {
        names = {"fast", "adaptive"};
        for (auto seed : seeds) names.push_back("random-" + std::to_string(seed));
    }
    std::vector<FlowEstimator> methods;
    for (const auto& name : names) methods.emplace_back(name.rfind("random-", 0) == 0 ? "adaptive" : name);
    std::cout << "split,scene,frame,method,median_ms,epe_sum,n,occ_sum,occ_n,boundary_sum,boundary_n";
    if (allocation) std::cout << ",eligible,checked";
    std::cout << '\n';
    int pairIndex = 0;
    std::string group, scene;
    int index;
    while (manifest >> group >> scene >> index) {
        if (group != split) continue;
        const auto p = split == "middlebury" ? loadMiddlebury(argv[1], scene, index)
                                            : loadPair(argv[1], scene, index);
        std::vector<std::vector<double>> times(names.size());
        std::vector<cv::Mat> outputs(names.size());
        std::vector<std::vector<uint8_t>> masks;
        RefinementSites quota;
        if (allocation) {
            methods[1].run(p.first, p.second);
            quota = methods[1].sites();
            for (auto seed : seeds) masks.push_back(randomSelection(quota, seed));
        }
        // Rotate methods both between repetitions and between pairs.
        for (int rep = -1; rep < repeats; ++rep) {
            for (size_t j = 0; j < names.size(); ++j) {
                const size_t m = (j + pairIndex + rep + 1) % names.size();
                const auto start = std::chrono::steady_clock::now();
                auto prediction = methods[m].run(p.first, p.second,
                                                  allocation && m >= 2 ? &masks[m - 2] : nullptr);
                const auto elapsed = std::chrono::steady_clock::now() - start;
                if (allocation && m > 0 && (methods[m].sites().checked != quota.checked ||
                                             methods[m].sites().eligible != quota.eligible))
                    throw std::runtime_error("selection budget differs between methods");
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
            if (allocation) std::cout << ',' << methods[m].sites().eligible << ',' << methods[m].sites().checked;
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
