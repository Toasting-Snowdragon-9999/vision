#ifndef CUDA_DEFINES_H
#define CUDA_DEFINES_H

namespace hsv {
    enum Colors {    
        /*
        Image from iphone 
        from python script converter:
        OpenCV HSV dark: (26.5, 255.0, 109.64999999999999)
        OpenCV HSV light: (28.5, 198.9, 247.35)
        */
        itennis_dark_hue = 26,
        itennis_dark_sat = 198,
        itennis_dark_val = 109,

        itennis_light_hue = 28,
        itennis_light_sat = 255,
        itennis_light_val = 255, 
        /*
        Image from webcame 
        from python script converter:
        OpenCV HSV dark: (46.0, 51.0, 175.95)
        OpenCV HSV light: (32.5, 25.5, 232.05)
        */
        wtennis_dark_hue = 46,
        wtennis_dark_sat = 51,
        wtennis_dark_val = 225,

        wtennis_light_hue = 35,
        wtennis_light_sat = 25,
        wtennis_light_val = 175,
    };
}

namespace brg {
    enum Colors {
        baby_pink_blue = 205,
        baby_pink_green = 150,
        baby_pink_red = 255,
    };
}

namespace two_dim {
    struct points{
        float x;
        float y;
    };

    struct sizes{
        float width;
        float height;
    };

    struct tracking_offset{
        float x_offset;
        float y_offset;
        float angle;
        float distance;
    };
}

#endif
