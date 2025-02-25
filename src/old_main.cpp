
#include <iostream>
#include <vector>

// Include OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include "cuda_defines.h"

std::vector<cv::Scalar> get_thresholds()
{
    std::vector<cv::Scalar> thresholds;

    std::vector<int> tolerane = {4, 15, 20};
    cv::Scalar lower_yellow(hsv::wtennis_light_hue - tolerane[0], hsv::wtennis_light_sat - tolerane[1], hsv::wtennis_light_val -  tolerane[2]);   // Lower HSV threshold
    cv::Scalar upper_yellow(hsv::wtennis_dark_hue + tolerane[0], hsv::wtennis_dark_sat + tolerane[2], hsv::wtennis_dark_val); // Upper HSV threshold

    thresholds.push_back(lower_yellow);
    thresholds.push_back(upper_yellow);
    return thresholds;
}

bool save_image(const cv::Mat& image, const std::string& filename) {
    if (image.empty()) {
        std::cerr << "Error: Empty image, nothing to save!" << std::endl;
        return false;
    }
    
    bool success = cv::imwrite(filename, image);
    
    if (success) {
        std::cout << "Image saved successfully to " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not save image to " << filename << std::endl;
    }
    
    return success;
}

cv::Mat load_image(const std::string& file_name) {
    cv::Mat image = cv::imread(file_name); 
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
    }
    return image;
}

cv::cuda::GpuMat image_processing(cv::cuda::GpuMat& src_host, std::vector<cv::Scalar>& cv_threshhold)
{
    try
    {
        cv::cuda::GpuMat d_hsv, d_mask, empty;
  
        cv::cuda::cvtColor(src_host, d_hsv, cv::COLOR_BGR2HSV);        

        cv::cuda::inRange(d_hsv, cv_threshhold[0], cv_threshhold[1], d_mask);

        return d_mask;
    }
    catch (const cv::Exception& ex)
    {
        std::cerr << "OpenCV error in image_processing: " << ex.what() << std::endl;
        cv::cuda::GpuMat empty;
        return empty;
    }
}


void display_image(const cv::Mat& image)
{
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);
}

void display_image_cuda(const cv::cuda::GpuMat& cuda_image)
{
    cv::Mat image;
    cuda_image.download(image);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);
}


cv::cuda::GpuMat gaussian_blur(const cv::cuda::GpuMat& d_src)
{
    cv::cuda::GpuMat d_blurred;
    try
    {
        cv::Ptr<cv::cuda::Filter> gaussFilter = cv::cuda::createGaussianFilter(
            d_src.type(),       // source type
            d_src.type(),       // destination type
            cv::Size(15, 15),   // kernel size
            0                   // sigmaX
            // (sigmaY defaults to 0 -> same as sigmaX)
        );

        gaussFilter->apply(d_src, d_blurred);
    }
    catch (const cv::Exception& ex)
    {
        std::cerr << "OpenCV error in gaussian_blur: " << ex.what() << std::endl;
    }
    return d_blurred;
}

void draw_square(cv::Mat& src_image, const cv::Mat& mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double min_contour_area = 100.0;
    double max_area = 0.0;
    int max_index = -1;

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        
        if (area < min_contour_area) {
            continue; // skips the rest of the code and starts the next iteration
        }

        if (area > max_area)
        {
            // find the largest contour area in the image
            max_area = area;
            max_index = static_cast<int>(i);
        }
    }

    if (max_index >= 0)
    {
        cv::Rect rect = cv::boundingRect(contours[max_index]);  
        
        int center_x = rect.x + rect.width / 2;
        int center_y = rect.y + rect.height / 2;
        // std::cout << "Center: (" << center_x << ", " << center_y << ")" << std::endl;
        std::cout << "Area: " << max_area << std::endl;
        rect.x      = std::max(0, rect.x);
        rect.y      = std::max(0, rect.y);
        rect.width  = std::min(rect.width, src_image.cols - rect.x);
        rect.height = std::min(rect.height, src_image.rows - rect.y);

        int thickness = 3;
        cv::rectangle(src_image,
                    rect,
                    cv::Scalar(brg::baby_pink_blue,
                                brg::baby_pink_green,
                                brg::baby_pink_red),
                    thickness);

        cv::Point origin(center_x, center_y);
        cv::circle(src_image,
                origin,
                1,                       // radius
                cv::Scalar(0, 0, 0),     // BGR: black
                cv::FILLED);
    }

    /*
    THIS IS TO MAKE A SQUARE INSTEAD OF A RECTANGLE
    */

    // if (max_index >= 0)
    // {
       
    //     cv::Rect rect = cv::boundingRect(contours[max_index]);  // calculates the smallest axis-aligned rectangle that completely covers the contour

    //     int side = std::max(rect.width, rect.height); // force square, by finding the largest side and make both that size

    //     int cx = rect.x + rect.width / 2;
    //     int cy = rect.y + rect.height / 2;
        
    //     std::cout << "Center: (" << cx << ", " << cy << ")" << std::endl;
        

    //     int half_side = side / 2;

    //     // Calculate top-left and bottom-right points
    //     // (Clamped to the image boundaries if necessary)
    //     cv::Point top_left(cx - half_side, cy - half_side);
    //     cv::Point bottom_right(cx + half_side, cy + half_side);

    //     // Ensure the square is within image bounds (optional safety check)
    //     top_left.x      = std::max(top_left.x, 0);
    //     top_left.y      = std::max(top_left.y, 0);
    //     bottom_right.x  = std::min(bottom_right.x, src_image.cols - 1);
    //     bottom_right.y  = std::min(bottom_right.y, src_image.rows - 1);

    //     int thickness = 3;

    //     cv::rectangle(src_image, top_left, bottom_right,  cv::Scalar(brg::baby_pink_blue, brg::baby_pink_green, brg::baby_pink_red), thickness);

    //     cv::Point origin(cx, cy);
    //     int center_r = 1;
    //     cv::circle(
    //         src_image,
    //         origin,             
    //         center_r,                      
    //         cv::Scalar(0, 0, 0),     // BGR: black
    //         cv::FILLED               // thickness or FILLED to make it a solid dot
    //     );
    // }
}

void draw_center_dot(cv::Mat &src, std::vector<int> size){
    int center_x = size[0] / 2;
    int center_y = size[1] / 2;
    int center_r = size[2];

    cv::circle(
        src,
        cv::Point(center_x, center_y),
        center_r,                    
        cv::Scalar(0, 0, 0),     // BGR: black
        cv::FILLED               // thickness or FILLED to make it a solid dot
    );
}


int main() {
    // Open camera (CPU capture)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera!" << std::endl;
        return -1;
    }

    cv::namedWindow("GPU Accelerated", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, blurred_cpu;         
    cv::cuda::GpuMat gpu_frame, grey_frame, d_blurred;  
    std::vector<cv::Scalar> threshholds = get_thresholds();
    for (;;) { 
        // infinite loop
        cap >> frame;         
        if (frame.empty()) {
            std::cerr << "Error: Empty frame!" << std::endl;
            break;
        }
        int radius = 1;
        std::vector <int> size = {frame.cols, frame.rows, radius};
        draw_center_dot(frame, size);

        gpu_frame.upload(frame);

        grey_frame = image_processing(gpu_frame, threshholds);

        d_blurred = gaussian_blur(grey_frame);



        d_blurred.download(blurred_cpu);
    
        draw_square(frame, blurred_cpu);

        cv::imshow("GPU Accelerated", frame);

        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    return 0;
}