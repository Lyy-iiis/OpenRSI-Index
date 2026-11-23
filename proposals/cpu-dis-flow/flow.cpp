#include "flow.hpp"

#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <stdexcept>

namespace {
constexpr int gridStep = 4;
constexpr int neighborDistance = 8;
constexpr int patchRadius = 2;
constexpr int residualThreshold = 8;
constexpr int minimumCostGain = 12;

std::array<cv::Point, 4> offsets(int distance) {
    return {cv::Point(-distance, 0), cv::Point(distance, 0),
            cv::Point(0, -distance), cv::Point(0, distance)};
}

cv::Point warp(cv::Point source, const cv::Vec2f& motion) {
    return source + cv::Point(cvRound(motion[0]), cvRound(motion[1]));
}

int intensityError(const cv::Mat& first, const cv::Mat& second, cv::Point a, cv::Point b) {
    return std::abs(first.at<uchar>(a) - second.at<uchar>(b));
}

int patchCost(const cv::Mat& first, const cv::Mat& second, cv::Point a, cv::Point b) {
    int cost = 2 * intensityError(first, second, a, b);
    for (const auto& offset : offsets(patchRadius))
        cost += intensityError(first, second, a + offset, b + offset);
    return cost;
}

// Reuse a nearby motion hypothesis if it explains a five-point patch better.
// Read every proposal from the original flow, then apply the corrections once.
RefinementSites refine(const cv::Mat& first, const cv::Mat& second, cv::Mat& flow,
                       bool gated, const std::vector<uint8_t>* selection) {
    RefinementSites sites;
    const cv::Rect validTarget(patchRadius, patchRadius,
                               second.cols - 2 * patchRadius, second.rows - 2 * patchRadius);
    cv::Mat correction((first.rows + gridStep - 1) / gridStep,
                       (first.cols + gridStep - 1) / gridStep, CV_32FC2, cv::Scalar(0, 0));
    const int start = neighborDistance + gridStep / 2;
    for (int y = start; y < first.rows - neighborDistance; y += gridStep) {
        for (int x = start; x < first.cols - neighborDistance; x += gridStep) {
            const cv::Point source(x, y);
            const auto current = flow.at<cv::Vec2f>(source);
            const auto target = warp(source, current);
            if (!validTarget.contains(target)) continue;
            const size_t index = sites.eligible++;
            if (selection) {
                if (!selection->at(index)) continue;
            } else if (gated && intensityError(first, second, source, target) < residualThreshold) {
                continue;
            }
            ++sites.checked;

            int bestCost = patchCost(first, second, source, target) - minimumCostGain;
            cv::Vec2f delta(0, 0);
            for (const auto& offset : offsets(neighborDistance)) {
                const auto proposed = flow.at<cv::Vec2f>(source + offset);
                const auto match = warp(source, proposed);
                if (match == target || !validTarget.contains(match)) continue;
                const int cost = patchCost(first, second, source, match);
                if (cost < bestCost) {
                    bestCost = cost;
                    delta = proposed - current;
                }
            }
            correction.at<cv::Vec2f>(y / gridStep, x / gridStep) = delta;
        }
    }
    cv::Mat dense;
    cv::resize(correction, dense, {}, gridStep, gridStep, cv::INTER_LINEAR);
    flow += dense(cv::Rect(0, 0, flow.cols, flow.rows));
    return sites;
}
}

FlowEstimator::FlowEstimator(const std::string& method) : method_(method) {
    if (method != "fast" && method != "medium" && method != "extra-iteration" &&
        method != "full-grid" && method != "adaptive")
        throw std::invalid_argument("unknown flow method: " + method);
    const auto preset = method == "medium" ? cv::DISOpticalFlow::PRESET_MEDIUM
                                           : cv::DISOpticalFlow::PRESET_FAST;
    dis_ = cv::DISOpticalFlow::create(preset);
    if (method == "extra-iteration")
        dis_->setVariationalRefinementIterations(dis_->getVariationalRefinementIterations() + 1);
}

cv::Mat FlowEstimator::run(const cv::Mat& first, const cv::Mat& second,
                           const std::vector<uint8_t>* selection) {
    cv::Mat flow;
    dis_->calc(first, second, flow);
    if (method_ == "adaptive" || method_ == "full-grid")
        sites_ = refine(first, second, flow, method_ == "adaptive", selection);
    return flow;
}
