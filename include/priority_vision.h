#ifndef PRIORITY_VISION_H
#define PRIORITY_VISION_H

#include "vision.h"

#define MAX_PRIOS 3

enum class Priority {
    LOW = 0,
    MEDIUM = 1,  
    HIGH = 2
};

struct ColorThreshold {
    std::vector<cv::Scalar> values;
    Priority priority;
    std::vector<std::vector<cv::Point>> contours;
};


class PrioVision : public Vision {
    public:
        PrioVision();
        ~PrioVision();
        void tracking () override;

    private: 
        std::vector<ColorThreshold> _thresholds;
        
        void _priorities();
        void _set_priority();

};   

#endif