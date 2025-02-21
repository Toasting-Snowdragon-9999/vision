#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

class Vision {
    public:
        Vision();
        ~Vision();
        void load_image(std::string image);
        void process_image();
        void detect_edges();
        void display_image();
        void detect_shape();

    private:
        cv::Mat _image;
        cv::Mat _gray;
        cv::Mat _edges;
        std::vector<std::vector<cv::Point>> _contours;
};
