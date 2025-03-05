#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>


void tracking(int camera_index) {
    int radius = 1;
    cv::Mat _image;

    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << camera_index << "!" << std::endl;
        return;
    }

    cv::namedWindow("GPU Accelerated", cv::WINDOW_AUTOSIZE);

    for (;;) { 
        cap >> _image;         
        if (_image.empty()) {
            std::cerr << "Error: Empty frame!" << std::endl;
            break;
        }

        // Get the center of the image
        int center_x = _image.cols / 2;
        int center_y = _image.rows / 2;

        // Convert to HSV
        cv::Mat hsv_image;
        cv::cvtColor(_image, hsv_image, cv::COLOR_BGR2HSV);

        // Get the HSV value at the center pixel
        cv::Vec3b hsv_pixel = hsv_image.at<cv::Vec3b>(center_y, center_x);

        int h = hsv_pixel[0]; // Hue
        int s = hsv_pixel[1]; // Saturation
        int v = hsv_pixel[2]; // Value

        std::cout << "Center HSV: (" << h << ", " << s << ", " << v << ")" << std::endl;

        // Draw a small circle at the center
        cv::circle(_image, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 0), -1);

        cv::imshow("GPU Accelerated", _image);
        
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
}

int main(){
    tracking(4);
    return 0;
}
