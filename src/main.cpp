
#include <iostream>
#include "opencv2/opencv.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>


cv::Mat image_processing(std::string file_name){
  
  cv::Mat empty; 
  
  try
    {
       
      // Load grayscale image (fixed the deprecated flag)
        cv::Mat src_host = cv::imread(file_name);
        if (src_host.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
            return empty;
        }

        // Use the correct CUDA namespace (cv::cuda)
        cv::cuda::GpuMat src, hsv, gpu_mask;
        src.upload(src_host);
    
        cv::cuda::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
        // Apply CUDA threshold
        
        // Define HSV range for tennis ball
        cv::Scalar lower_yellow(25, 80, 80);   // Lower HSV threshold
        cv::Scalar upper_yellow(35, 255, 255); // Upper HSV threshold

        // Apply thresholding in HSV color space (CUDA)
        cv::cuda::inRange(hsv, lower_yellow, upper_yellow, gpu_mask);        // Download back to CPU memory
        cv::Mat mask;
        gpu_mask.download(mask);
        return mask;
   }
    catch(const cv::Exception& ex)
    {
     
   std::cerr << "Error: " << ex.what() << std::endl;
        return empty;
    }
}

void display_image(cv::Mat image){
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
  cv::imshow("Display Image", image);
  cv::waitKey(0);
}

cv::Mat draw_circle(cv::Mat src_image, cv::cuda::GpuMat blurred){
  cv::Mat src = src_image;
  std::vector<cv::Vec3f> circles;
  cv::Mat blurred_cpu;
  blurred.download(blurred_cpu);
  cv::HoughCircles(blurred_cpu, circles, cv::HOUGH_GRADIENT, 1, 20, 50, 30, 20, 40);
  for (size_t i = 0; i < circles.size(); i++)
  {
    cv::Vec3f c = circles[i];
    cv::Point center(cvRound(c[0]), cvRound(c[1]));
    int radius = cvRound(c[2]);
    cv::circle(src, center, radius, cv::Scalar(0, 255, 0), 4);
    cv::circle(src, center, 5, cv::Scalar(0, 0, 255), -1);
  }
  return src;
}


cv::cuda::GpuMat gaussian_blur(cv::Mat image){
    cv::cuda::GpuMat blurred_image;
    cv::GaussianBlur(image, blurred_image, cv::Size(15, 15), 0); // Use regular cv::GaussianBlur here
    return blurred_image;
}


int main (int argc, char* argv[])
{

  std::string file_name = "..//pics/tennis_ball.jpg";
  std::string file_name2 = "..//pics/man_holding_tennis_ball4.jpg";
  std::string file_name3 = "..//pics/tennis_ball_nature.jpg";
  cv::Mat grey_image = image_processing(file_name3);
  cv::cuda::GpuMat blurred = gaussian_blur(grey_image);
  cv::Mat image = draw_circle(grey_image, blurred);
  display_image(image);
    


  return 0;
}

