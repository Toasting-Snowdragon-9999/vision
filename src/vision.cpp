#include "vision.h"

Vision::Vision() {
    std::cout << "Vision object created" << std::endl;
}

Vision::~Vision() {
    std::cout << "Vision object destroyed" << std::endl;
}

void Vision::load_image(std::string image) {
    _image = cv::imread(image);
    if (_image.empty()) {
        std::cerr << "Could not read the image: " << image << std::endl;
    }
}

void Vision::process_image() {
    cv::cvtColor(_image, _gray, cv::COLOR_BGR2GRAY);
}
