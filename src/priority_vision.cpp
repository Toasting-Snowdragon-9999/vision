#include "priority_vision.h"

PrioVision::PrioVision() {

}

PrioVision::~PrioVision() {

}

void PrioVision::tracking() {
    int radius = 1;
    two_dim::tracking_offset offset;

    cv::VideoCapture cap(0);
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

        for(int i = 0; i < MAX_PRIOS; i++){
            _greyscale_image(_gpu_frame, _thresholds[i].values);        // find the contours for each color
            _thresholds[i].contours = _contours;
        }

        /**
         * Make a loop of some kind that (maybe switch case) that takes the highest
         * priority and tracks their contour, its not _greyscale_image but _draw_rect 
         * that should be called or changed so that the contours are returned and store.
         */

        _gaussian_blur(_grey);

        _blurred.download(_blurred_cpu);
    
        _draw_rect(_image, _blurred_cpu);

        _calculate_offset(offset);

        cv::imshow("GPU Accelerated", _image);
        
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }


}
