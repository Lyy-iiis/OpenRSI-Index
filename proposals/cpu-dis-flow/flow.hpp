#pragma once

#include <opencv2/video/tracking.hpp>
#include <cstdint>
#include <vector>

struct RefinementSites {
    size_t eligible = 0;
    size_t checked = 0;
};

// All methods receive one pair of grayscale frames, with no video history.
class FlowEstimator {
public:
    explicit FlowEstimator(const std::string& method);
    cv::Mat run(const cv::Mat& first, const cv::Mat& second,
                const std::vector<uint8_t>* selection = nullptr);
    RefinementSites sites() const { return sites_; }

private:
    RefinementSites sites_;
    std::string method_;
    cv::Ptr<cv::DISOpticalFlow> dis_;
};
