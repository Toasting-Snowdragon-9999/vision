#include "vision.h"

Vision::Vision() {
    std::cout << "Vision object created" << std::endl;
}

Vision::~Vision() {
    std::cout << "Vision object destroyed" << std::endl;
}

void Vision::load_image(const std::string& file_name) 
{
    _image = cv::imread(file_name); 
    if (_image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
    }
}

void Vision::display_image(const cv::Mat& image)
{
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);
}

bool Vision::save_image(const cv::Mat& image, const std::string& filename) 
{
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

void Vision::tracking() {
    bool enable_calibration = true;    
    int radius = 1;
    two_dim::tracking_offset offset;
    std::vector<cv::Scalar> threshholds = _get_thresholds(enable_calibration);
    std::cout << "thresholds: " << threshholds[0] << " " << threshholds[1] << std::endl;

    cv::VideoCapture cap(EXTERNAL_CAMERA);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera!" << std::endl;
        return;
    }
    cv::namedWindow("GPU Accelerated", cv::WINDOW_AUTOSIZE);
    
    for (;;) { 
        // infinite loop
        cap >> _image;         
        if (_image.empty()) {
            std::cerr << "Error: Empty frame!" << std::endl;
            break;
        }

        std::vector <int> size = {_image.cols, _image.rows, radius};
        _draw_center_dot(_image, size);
        
        _mark_cornors(_image);

        _gpu_frame.upload(_image);

        _greyscale_image(_gpu_frame, threshholds);

        _gaussian_blur(_grey);

        _blurred.download(_blurred_cpu);

        _find_contours(_contours, _blurred_cpu);
    
        _draw_rect(_image, _contours);

        _calculate_offset(offset);

        cv::imshow("GPU Accelerated", _image);
        
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

}


/*
These are the helper functions to make the complete tracking using OpenCV and CUDA.
*/

void Vision::_calculate_offset(two_dim::tracking_offset& output)
{
    output.x_offset = _center_image.x - _center_tracking.x;
    output.y_offset = _center_image.y - _center_tracking.y;

    output.distance = std::sqrt(output.x_offset * output.x_offset + output.y_offset * output.y_offset);
    
    output.angle = std::atan2(output.y_offset, output.x_offset) * 180.0 / CV_PI;

    // std::cout << "Offset: (" << output.x_offset << ", " << output.y_offset << ")" << std::endl;
    // std::cout << "Distance: " << output.distance << std::endl;
    // std::cout << "Angle: " << output.angle << std::endl;
}


std::vector<cv::Scalar> Vision::_get_thresholds(bool run_calibration)
{
    std::vector<cv::Scalar> thresholds;

    if(run_calibration){
        _calibration(thresholds);
    }
    else
    {

        std::vector<int> tolerane = {3, 15, 20, 0};
        cv::Scalar lower_bound(
            hsv::wtennis_light_hue - tolerane[0], 
            hsv::wtennis_light_sat - tolerane[3], 
            hsv::wtennis_light_val -  tolerane[2]);   // Lower HSV threshold
        
        cv::Scalar upper_bound(
            hsv::wtennis_dark_hue + tolerane[0], 
            hsv::wtennis_dark_sat + tolerane[2], 
            hsv::wtennis_dark_val - tolerane[3]); // Upper HSV threshold

        thresholds.push_back(lower_bound);
        thresholds.push_back(upper_bound);
        thresholds = hsv::predefines_yellow();
    }

    return thresholds;
}

void Vision::_greyscale_image(cv::cuda::GpuMat& src_host, std::vector<cv::Scalar>& cv_threshhold) 
{
    try
    {
        cv::cuda::cvtColor(src_host, _hsv, cv::COLOR_BGR2HSV);        
        cv::cuda::inRange(_hsv, cv_threshhold[0], cv_threshhold[1], _grey);
    }
    catch (const cv::Exception& ex)
    {
        std::cerr << "OpenCV error in image_processing: " << ex.what() << std::endl;
        cv::cuda::GpuMat empty;
    }

}

void Vision::_gaussian_blur(const cv::cuda::GpuMat& d_src)
{
    try
    {
        cv::Ptr<cv::cuda::Filter> gaussFilter = cv::cuda::createGaussianFilter(
            d_src.type(),       // source type
            d_src.type(),       // destination type
            cv::Size(15, 15),   // kernel size
            0                   // sigmaX
            // (sigmaY defaults to 0 -> same as sigmaX)
        );

        gaussFilter->apply(d_src, _blurred);
    }
    catch (const cv::Exception& ex)
    {
        std::cerr << "OpenCV error in gaussian_blur: " << ex.what() << std::endl;
    }

}

void Vision::_find_contours(std::vector<std::vector<cv::Point>>& contours, const cv::Mat& mask){
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}

void Vision::_draw_rect(cv::Mat& src_image, std::vector<std::vector<cv::Point>>& contours)
{
    // std::vector<std::vector<cv::Point>> contours;
    // cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double min_contour_area = 400.0;
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
        // std::cout << "Area: " << max_area << std::endl;
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
        _center_tracking.x = center_x;
        _center_tracking.y = center_y;
        cv::circle(src_image,
                origin,
                1,                       // radius
                cv::Scalar(0, 0, 0),     // BGR: black
                cv::FILLED);
    }

}



void Vision::_draw_square(cv::Mat& src_image, std::vector<std::vector<cv::Point>>& contours)
{
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
       
        cv::Rect rect = cv::boundingRect(contours[max_index]);  // calculates the smallest axis-aligned rectangle that completely covers the contour

        int side = std::max(rect.width, rect.height); // force square, by finding the largest side and make both that size

        int center_x = rect.x + rect.width / 2;
        int center_y = rect.y + rect.height / 2;
        
        // std::cout << "Center: (" << center_x << ", " << center_y << ")" << std::endl;
        

        int half_side = side / 2;

        // Calculate top-left and bottom-right points
        // (Clamped to the image boundaries if necessary)
        cv::Point top_left(center_x - half_side, center_y - half_side);
        cv::Point bottom_right(center_x + half_side, center_y + half_side);

        // Ensure the square is within image bounds (optional safety check)
        top_left.x      = std::max(top_left.x, 0);
        top_left.y      = std::max(top_left.y, 0);
        bottom_right.x  = std::min(bottom_right.x, src_image.cols - 1);
        bottom_right.y  = std::min(bottom_right.y, src_image.rows - 1);

        int thickness = 3;

        cv::rectangle(src_image, top_left, bottom_right,  cv::Scalar(brg::baby_pink_blue, brg::baby_pink_green, brg::baby_pink_red), thickness);

        cv::Point origin(center_x, center_y);
        _center_tracking.x = center_x;
        _center_tracking.y = center_y;
        int center_r = 1;
        cv::circle(
            src_image,
            origin,             
            center_r,                      
            cv::Scalar(0, 0, 0),     // BGR: black
            cv::FILLED               // thickness or FILLED to make it a solid dot
        );
    }
}

void Vision::_draw_center_dot(cv::Mat &src, std::vector<int> size){
    int center_x = size[0] / 2;
    int center_y = size[1] / 2;
    int center_r = size[2];
    _center_image.x = center_x;
    _center_image.y = center_y;

    cv::circle(
        src,
        cv::Point(center_x, center_y),
        center_r,                    
        cv::Scalar(0, 0, 0),     // BGR: black
        cv::FILLED               // thickness or FILLED to make it a solid dot
    );
}

void Vision::_draw_dot(cv::Mat &src, int x, int y, cv::Vec3b color){
    src.at<cv::Vec3b>(y, x) = color; // red pixel
}

void Vision::_mark_cornors(cv::Mat &src){
    cv::Vec3b blue(255, 0, 0);
    cv::Vec3b red(0, 0, 255);
    cv::Vec3b green(0, 255, 0);
    cv::Vec3b purple(255, 0, 255);


    int width = src.cols;
    int height = src.rows;
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++)
        {
            _draw_dot(src, 0 + i, 0 + j, blue);                     // top left
            _draw_dot(src, width - 1 - i, 0 + j, red);              // top right
            _draw_dot(src, 0 + i, height - 1 - j, green);           // bottom left
            _draw_dot(src, width - 1 - i, height - 1 - j, purple);  // bottom right
        }
    }

}

void Vision::_calibration(std::vector<cv::Scalar>& thresholds) {
    std::cout << "Calibration" << std::endl;
    std::vector<cv::Vec3b> center_points(MAX_CALIBRATION);
    cv::VideoCapture cap(EXTERNAL_CAMERA);
    int i = 0;
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera!" << std::endl;
        return;
    }
    cv::namedWindow("Calibration", cv::WINDOW_AUTOSIZE);
    
    for (;;) { 
        cap >> _image;         
        if (_image.empty()) {
            std::cerr << "Error: Empty frame!" << std::endl;
            break;
        }

        int center_x = _image.cols / 2;
        int center_y = _image.rows / 2;

        cv::Mat hsv_image;
        cv::cvtColor(_image, hsv_image, cv::COLOR_BGR2HSV);

        cv::Vec3b hsv_pixel = hsv_image.at<cv::Vec3b>(center_y, center_x);
        center_points[i] = hsv_pixel;
        i++;
        if (i == MAX_CALIBRATION){
            i = RESET;
        }
        cv::circle(_image, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 0), -1);

        cv::imshow("GPU Accelerated", _image);
        
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    double avg_hue = RESET;
    double avg_sat = RESET;
    double avg_val = RESET;
    double lowest_hue = MAX_8_BIT;
    double lowest_sat = MAX_8_BIT;
    double lowest_val = MAX_8_BIT;
    double highest_hue = RESET;
    double highest_sat = RESET;
    double highest_val = RESET;

    for (int i = 0; i < MAX_CALIBRATION; i++){
        if (center_points[i][0] < lowest_hue){
            lowest_hue = center_points[i][0];
        }
        if (center_points[i][1] < lowest_sat){
            lowest_sat = center_points[i][1];
        }
        if (center_points[i][2] < lowest_val){
            lowest_val = center_points[i][2];
        }
        if (center_points[i][0] > highest_hue){
            highest_hue = center_points[i][0];
        }
        if (center_points[i][1] > highest_sat){
            highest_sat = center_points[i][1];
        }
        if (center_points[i][2] > highest_val){
            highest_val = center_points[i][2];
        }

        avg_hue += center_points[i][0];
        avg_sat += center_points[i][1];
        avg_val += center_points[i][2];
    }
    avg_hue = avg_hue / MAX_CALIBRATION;
    avg_sat = avg_sat / MAX_CALIBRATION;
    avg_val = avg_val / MAX_CALIBRATION;


    std::vector<int> thresholds_values = {5, 30, 30}; // hue, sat, val
    cv::Scalar lower_bound(avg_hue - thresholds_values[0], avg_sat - thresholds_values[1], avg_val - thresholds_values[2]);   // Lower HSV threshold
    cv::Scalar upper_bound(avg_hue + thresholds_values[0], avg_sat + thresholds_values[1], avg_val + thresholds_values[2]); // Upper HSV threshold
    std::cout << "Lower bound: " << lower_bound << std::endl;
    std::cout << "Upper bound: " << upper_bound << std::endl;
    
    thresholds.push_back(lower_bound);
    thresholds.push_back(upper_bound);
    cv::destroyWindow("Calibration");

}