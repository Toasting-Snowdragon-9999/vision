#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include "cuda_defines.h"

class Vision {
    public:
        Vision();
        ~Vision();
        void tracking();
        void load_image(const std::string& file_name);
        bool save_image(const cv::Mat& image, const std::string& filename);
        void display_image(const cv::Mat& image);
    
    private:
        cv::Mat _image, _blurred_cpu;
        cv::cuda::GpuMat _grey, _hsv, _blurred, _gpu_frame;

        std::vector<std::vector<cv::Point>> _contours;

        std::vector<cv::Scalar> _get_thresholds();
        void _greyscale_image(cv::cuda::GpuMat& src_host, std::vector<cv::Scalar>& cv_threshhold);
        void _gaussian_blur(const cv::cuda::GpuMat& d_src);
        void _draw_rect(cv::Mat& src_image, const cv::Mat& mask);
        void _draw_square(cv::Mat& src_image, const cv::Mat& mask);
        void _draw_center_dot(cv::Mat &src, std::vector<int> size);

};
