#pragma once

#include <opencv2/video/tracking.hpp>

// All methods receive one pair of grayscale frames, with no video history.
class FlowEstimator {
public:
    explicit FlowEstimator(const std::string& method);
    cv::Mat run(const cv::Mat& first, const cv::Mat& second);

private:
    std::string method_;
    cv::Ptr<cv::DISOpticalFlow> dis_;
};
