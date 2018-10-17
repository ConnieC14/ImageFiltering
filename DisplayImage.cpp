/*
 * DisplayImage.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: Connie
 */
#include <opencv2/core/core_c.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <highgui.h>
//#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

/** @function that checks if number is an int */
bool isInteger(double a){
    double b=round(a),epsilon=1e-9; //some small range of error
    return (a<=b+epsilon && a>=b-epsilon);
}

Mat FixDimensions(Mat src, Mat dst, double delta) {
	if ( isInteger((double) src.cols*delta) && isInteger((double) src.rows*delta) ) {
		  if (delta > 1) {
			  resize(src, dst, Size(), delta, delta, CV_INTER_LINEAR);
		  }
		  if (delta == 1.0) {
		  		  return dst;
		  	  }
		  else {
			  resize(src, dst, Size(), delta, delta, CV_INTER_AREA);
		  }
	  }

	  else {
		  int test = (int) src.cols * delta;
		  int test2 = (int) src.rows * delta;

		  if (delta > 1) {
	  		  resize(src, dst, Size(), delta, delta, CV_INTER_LINEAR);
		  }
		  if (delta == 1.0) {
			  return dst;
		  }
		  else {
	  		  resize(src, dst, Size(), delta, delta, CV_INTER_AREA);
		  }
	  }
	return dst;
}

int StepSize(float delta) {
	int step = 0;
	(delta == 0) ? delta = 1 : delta;
		if (delta < 1) {
			step = (int)1 / delta;
		}
		else { step = (int)delta; }
		std::cout << "Step: " << step << endl;
		return step;
}

void CreateWindow(char* Sname, char* Dname, Mat source, Mat dest) {
	// window name
	char* window_name = Sname;
	char* window_name2 = Dname;

	// Create window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	namedWindow( window_name2, CV_WINDOW_AUTOSIZE );

	// Show result in a window
	imshow(window_name, source);
	imshow(window_name2, dest);
	// Create an image
	//imwrite( "new.jpg", dst);

	cout << "creating windows" << endl;

	// Press a key to close window
	waitKey(0);
}

void CreateWindow(char* Sname, Mat source) {
	// window name
	char* window_name = Sname;
	// Create window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	// Show result in a window
	imshow(window_name, source);

	// Create an image
	//imwrite( "new.jpg", dst);

	cout << "creating windows" << endl;

	// Press a key to close window
	waitKey(0);
}

void BoxFilter(Mat src, Mat dst, Mat kernel, int imgHeight, int imgWidth, int kHeight, int kWidth, double delta, int number) {
	/*****************************************************************/
	/// Update kernel size for a normalized box filter
	/** sample kernel:
	  			 *  | 1/9 1/9 1/9 |
	  			 *  | 1/9 1/9 1/9 |
	  			 *  | 1/9 1/9 1/9 |  */
	kernel = Mat::ones(kHeight, kWidth, CV_32F) / (float)(kHeight*kWidth);

	// Determining downsampling step size
	int step;
	(delta == 0) ? delta=1 : delta;
	if (delta < 1 ){
	step = (int) 1/delta; }
	else{ step = (int) delta;}
	cout << "Step: " << step << endl;
    /*****************************************************************/
	// initializing Box Filter
	for (int x_col=0; x_col < imgWidth; x_col+=step) {//x
	    for(int y_row=0; y_row < imgHeight; y_row+=step) {//y
		  // initializing colors
		  float red = 0.0, green = 0.0, blue = 0.0;

		  //Kernel's center will be at (m,n)
		  for(int kernelX = 0;kernelX < kWidth;kernelX++) { // x
			for(int kernelY = 0; kernelY < kHeight; kernelY++) { // y
				float tempRed, tempGreen, tempBlue;
//				Vec3b color = src.at<Vec3b>(y_row,x_col);

				// calculate the pixel location for the image
				int imageX = (x_col - (int)(kWidth / 2) + kernelX);
				int imageY = (y_row - (int)(kHeight / 2) + kernelY);

				// if kernel is off
				if (imageY < 0 || imageX < 0) {
					tempBlue = 0;
					tempGreen = 0;
					tempRed = 0;
			     }

				else {
					// Colors are in BGR format
					tempBlue = src.at<Vec3b>(imageY, imageX).val[0];
					tempGreen = src.at<Vec3b>(imageY, imageX).val[1];
					tempRed = src.at<Vec3b>(imageY, imageX).val[2];
				}

			    //multiply every value of the filter with corresponding image pixel
			    blue += ( tempBlue * kernel.at<float>(kernelY, kernelX) );
				green += ( tempGreen * kernel.at<float>(kernelY, kernelX) );
				red += ( tempRed * kernel.at<float>(kernelY,kernelX) );
			}
		 }

		 // write in color results to the destination file
		 dst.at<Vec3b>(y_row * delta,x_col * delta).val[0] = blue;
		 dst.at<Vec3b>(y_row * delta,x_col * delta).val[1] = green;
		 dst.at<Vec3b>(y_row * delta,x_col * delta).val[2] = red;
	   }
	}
	cout << "done" << endl;
	string text = "BLURR";
	text += std::to_string(number);
	text += ".jpg";
	imwrite(text, dst);

}

void xyGaussianFilter(Mat src, Mat dst, Mat kernel, int imgHeight, int imgWidth, int kHeight, int kWidth, int step) {
	/*****************************************************************/
	// Initialize variables, sigma, radius squared, and normalization constant.
	float sigma = 1.0, sum = 0.0;
	float r_sq = 0, denom = 2 * sigma * sigma;
	float delta = 1.0 / step;
	// Update kernel size for 2D Gaussian Kernel
	kernel = Mat::ones(kHeight, kWidth, CV_32F);

	// Calculate bounds for for loop
	int lowXbound = -floor(kWidth / 2);
	int upXbound = ceil(kWidth / 2);
	int lowYbound = -floor(kHeight / 2);
	int upYbound = ceil(kHeight / 2);

	// Fill in the kernel values appropriately
	for (int xk = lowXbound; xk <= upXbound; xk++) {
		for (int yk = lowYbound; yk <= upYbound; yk++) {
			r_sq = xk * xk + yk * yk;
			kernel.at<float>(yk + 2, xk + 2) = exp(-(r_sq / denom)) * (1 / (M_PI * denom));
			sum += kernel.at<float>(yk + 2, xk + 2);
		}
	}

	for (int x = 0; x < kWidth; x++) {
		for (int y = 0; y < kHeight; y++) {
			kernel.at<float>(y, x)  /= sum;;
		}
	}

	// Downsampling process & make sure dimensions are integers
	cout << "downsampled image width: " << src.cols*delta << endl;
	cout << "downsampled image height: " << src.rows * delta << endl;

	std::cout << "Step: " << step << endl;
	/***************************************************************/
	// Initializing Gaussian Filter
	for (int x_col = 0; x_col < imgWidth; x_col += step) {//x
		for (int y_row = 0; y_row < imgHeight; y_row += step) {//y
			// Initializing colors
			float red = 0.0, green = 0.0, blue = 0.0;

			// Kernel's center will be at (m,n)
			for (int kernelX = 0; kernelX < kWidth; kernelX++) { // x
				for (int kernelY = 0; kernelY < kHeight; kernelY++) { // y
					float tempRed = 0.0, tempGreen = 0.0, tempBlue = 0.0;

					// Calculate the pixel location for the image
					int imageX = (x_col - (int)(kWidth / 2) + kernelX);
					int imageY = (y_row - (int)(kHeight / 2) + kernelY);

					// If kernel is off
					if (imageY < 0 || imageX < 0) {
						tempBlue = 0;
						tempGreen = 0;
						tempRed = 0;
					}

					else {
						// Colors are in BGR format
						tempBlue = src.at<Vec3b>(imageY, imageX).val[0];
						tempGreen = src.at<Vec3b>(imageY, imageX).val[1];
						tempRed = src.at<Vec3b>(imageY, imageX).val[2];
					}

					// Multiply every value of the filter with corresponding image pixel
					blue += (tempBlue * kernel.at<float>(kernelY, kernelX));
					green += (tempGreen * kernel.at<float>(kernelY, kernelX));
					red += (tempRed * kernel.at<float>(kernelY, kernelX));
				}
			}

			// Write in color results to the destination file
			cout << "hello" << endl;
			dst.at<Vec3b>(y_row * delta, x_col * delta).val[0] = blue;
			dst.at<Vec3b>(y_row * delta, x_col * delta).val[1] = green;
			dst.at<Vec3b>(y_row * delta, x_col * delta).val[2] = red;
		}
	}
	cout << "finished" << endl;
}

void zGaussianFilter(std::vector<Mat> src, Mat dst, Mat kernel, int imgHeight, int imgWidth, int kHeight, int kWidth, double delta) {
	/*****************************************************************/
	// Initialize variables, sigma, radius squared, and normalization constant.
	float sigma = 1.0, sum = 0.0;
	float r_sq, denom = 2 * sigma * sigma;

	// Update kernel size for 2D Gaussian Kernel
	kernel = Mat::ones(kHeight, kWidth, CV_32F);

	// Calculate bounds for for loop
	int lowXbound = -floor(kWidth / 2);
	int upXbound = ceil(kWidth / 2);
	int lowYbound = -floor(kHeight / 2);
	int upYbound = ceil(kHeight / 2);


	// Fill in the kernel values appropriately
	for (int xk = lowXbound; xk < upXbound; xk++) {
		for (int yk = lowYbound; yk < upYbound; yk++) {
			r_sq = xk * xk + yk * yk;
			kernel.at<float>(yk + 2, xk + 2) = exp(-(r_sq / denom)) * (1 / (M_PI * denom));
			sum += kernel.at<float>(yk + 2, xk + 2);
		}
	}

	for (int x = 0; x < kWidth; x++) {
		for (int y = 0; y < kHeight; y++) {
			kernel.at<float>(y, x) /= sum;
		}
	}

	// Determining downsampling step size
	int step;
	(delta == 0) ? delta = 1 : delta;
	if (delta < 1) {
		step = (int)1 / delta;
	}
	else { step = (int)delta; }
	std::cout << "Step: " << step << endl;
	/*****************************************************************/
	// Initializing Gaussian Filter
	for (int x_col = 0; x_col < imgWidth; x_col += step) {//x
		for (int y_row = 0; y_row < imgHeight; y_row += step) {//y
		// Initializing colors
			float red = 0.0, green = 0.0, blue = 0.0;

			//Kernel's center will be at (m,n)
			for (int kernelX = 0; kernelX < kWidth; kernelX++) { // x
				for (int kernelY = 0; kernelY < kHeight; kernelY++) { // y
					float tempRed, tempGreen, tempBlue;
					// Vec3b color = src.at<Vec3b>(y_row,x_col);

					// run through the array of images
					for (int i=0; i < src.size(); i++) {

						// calculate the pixel location for the image
						int imageX = (x_col - (int)(kWidth / 2) + kernelX);
						int imageY = (y_row - (int)(kHeight / 2) + kernelY);

						// if kernel is off
						if (imageY < 0 || imageX < 0) {
							tempBlue = 0;
							tempGreen = 0;
							tempRed = 0;
						}

						else {
							// Colors are in BGR format
							tempBlue = src[i].at<Vec3b>(imageY, imageX).val[0];
							tempGreen = src[i].at<Vec3b>(imageY, imageX).val[1];
							tempRed = src[i].at<Vec3b>(imageY, imageX).val[2];
						}

						//multiply every value of the filter with corresponding image pixel
						blue += (tempBlue * kernel.at<float>(kernelY, kernelX));
						green += (tempGreen * kernel.at<float>(kernelY, kernelX));
						red += (tempRed * kernel.at<float>(kernelY, kernelX));
					}
				}
			}

			// write in color results to the destination file
			dst.at<Vec3b>(y_row * delta, x_col * delta).val[0] = blue;
			dst.at<Vec3b>(y_row * delta, x_col * delta).val[1] = green;
			dst.at<Vec3b>(y_row * delta, x_col * delta).val[2] = red;
		}
	}
}


Mat xyDirectionalFilter(Mat src, Mat dst, int step, bool y_true, int interval_height, int interval_width) {
	/*****************************************************************/
	Mat X_kernel, Y_kernel, tempo = src;
	float avg_direction = 0.0, avg_magnitude = 0.0, angle = 0.0;
	int kHeight = 3, kWidth = 3, counter = 0;

	// Converting from color image (B, G, R) to grayscale
	cvtColor(src, src, COLOR_BGR2GRAY);

	// Fill temp with a color
	tempo = Scalar(59, 51, 58);

	// Update kernel size for a normalized box filter
	/** sample x-dir kernel:
	*  | -1 0 1 |
	*  | -1 0 1 |
	*  | -1 0 1 |  */
	X_kernel = (Mat_<float>(kHeight, kWidth) << -1, 0, 1,
		-1, 0, 1,
		-1, 0, 1);

	Y_kernel = X_kernel.t();
	std::cout << "transposed matrix: " << Y_kernel << endl;

	/*****************************************************************/
	// Initializing Convolution
	for (int x_col = 2; x_col < src.cols-1; x_col += step) {//x
		for (int y_row = 2; y_row < src.rows-1; y_row += step) {//y
			// initializing colors
			float xintensity = 0.0, yintensity = 0.0;

			// Kernel's center will be at (m,n)
			for (int kernelX = 0; kernelX < kWidth; kernelX++) { // x
				for (int kernelY = 0; kernelY < kHeight; kernelY++) { // y
					// Save intensity here
					float tempInt;

					// Calculate the pixel location for the image
					int imageX = (x_col - (int)(kWidth / 2) + kernelX);
					int imageY = (y_row - (int)(kHeight / 2) + kernelY);

					// If kernel is out of bounds
					if (imageY < 0 || imageX < 0) {
						tempInt = 0;
					}
					else {
						// Colors are in BGR format
						tempInt = src.at<uchar>(imageY, imageX );
					}

					// Multiply every value of the filter with corresponding image pixel
					float xkern = (X_kernel.at<float>(kernelY, kernelX));
					float ykern = (Y_kernel.at<float>(kernelY, kernelX));
					xintensity += ((tempInt)* xkern);
					yintensity += ((tempInt)* ykern);
				}
			}
			// Update variables and calculate magnitude and direction of vector
			float xcolor = abs(xintensity / 3);
			float ycolor = abs(yintensity / 3);
			int magnitude = sqrt(xcolor*xcolor + ycolor*ycolor);


			// Calculate the direction/angle of the vector
			if (xintensity != 0) {
				angle = atan((yintensity / 3) / (xintensity / 3));
			}
			else {
				if (yintensity > 0) {
					angle = M_PI / 2;
				}
				if (yintensity < 0) {
					angle = -M_PI / 2;
				}
			}

			// Add all values calculated
			avg_magnitude += magnitude;
			avg_direction += angle;
			counter++;

			if ((x_col % interval_width) == 0 && (y_row % interval_height) == 0) {

				// Find the average direction of all
				avg_direction = avg_direction / counter;
				avg_magnitude = avg_magnitude / counter;

				// Find the midpoint of grid and draw lines from there
				Point midpoint, pt1, pt2;
				midpoint.x = (x_col) + interval_width / 2;
				midpoint.y = (y_row) + interval_height / 2;

				/*********************************************************************************/
				// Calculate the points to connect and their slope using the angle
				pt2.y = (int)round(midpoint.y + (interval_width / 2) * cos(avg_direction));
				pt2.x = (int)round(midpoint.x + (interval_height / 2) * sin(avg_direction));

				pt1.y = (int)round(midpoint.y - (interval_width / 2) * cos(avg_direction));
				pt1.x = (int)round(midpoint.x - (interval_height / 2) * sin(avg_direction));

				// Conditions in case the points are off the image grid
				std::cout << "\t\t\t\t\t\t\t midpoint is: " << midpoint << endl;
				std::cout << "\t\t\t\t\t\t\t pt1: " << pt1 << " pt2: " << pt2 << endl;
				if ((pt1.x > 0 & pt1.x <= src.cols) && (pt1.y > 0 & pt1.y <= src.rows)
						&& (pt2.x > 0 & pt2.x <= src.cols ) && (pt2.y > 0 & pt2.y <= src.rows)) {
					// Now we want to draw a vector line
					arrowedLine(tempo, pt1, pt2, Scalar(130, avg_magnitude, 216), 4, CV_AA);
				}

				// Reset parameters
				avg_direction = 0;
				avg_magnitude = 0;
				counter = 0;
			}
		}
	}
	cout << "finished" << endl;
	return tempo;
}


Mat zDirectionalFilter(std::vector<Mat> src, Mat dst, int step, bool y_true, int interval_height, int interval_width) {
	/*****************************************************************/
	Mat X_kernel, Y_kernel, tempo = dst;
	float avg_direction, avg_magnitude, angle;
	int counter = 0, kHeight = 3, kWidth = 3;
	int imgWidth = src[0].cols , imgHeight = src[0].rows;

	// Converting from color image (B, G, R) to grayscale
	cvtColor(src, src, COLOR_BGR2GRAY);

	// Fill tempo with a color
	tempo = Scalar(59, 51, 58);

	// Update kernel size for a normalized box filter
	/** sample x-dir kernel:
	*  | -1 0 1 |
	*  | -1 0 1 |
	*  | -1 0 1 |  */
	X_kernel = (Mat_<float>(kHeight, kWidth) << -1, 0, 1,
		-1, 0, 1,
		-1, 0, 1);

	Y_kernel = X_kernel.t();

	/*****************************************************************/
	// Initializing Convolution
	for (int x_col = 0; x_col < imgWidth-1; x_col += step) {//x
		for (int y_row = 0; y_row < imgHeight-1; y_row += step) {//y

			for (int i = 0; i < src.size(); i++) {												   // initializing colors
				float xintensity = 0.0, yintensity = 0.0;

			// Kernel's center will be at (m,n)
			for (int kernelX = 0; kernelX < kWidth; kernelX++) { // x
				for (int kernelY = 0; kernelY < kHeight; kernelY++) { // y
					float tempInt;

						// Calculate the pixel location for the image
						int imageX = (x_col - (int)(kWidth / 2) + kernelX);
						int imageY = (y_row - (int)(kHeight / 2) + kernelY);

						// If kernel is out of bounds
						if (imageY < 0 || imageX < 0) {
							tempInt = 0;
						}
						else {
							// Colors are in BGR format
							tempInt = src[i].at<uchar>(imageY, imageX);
						}

						// Multiply every value of the filter with corresponding image pixel
						float xkern = (X_kernel.at<float>(kernelY, kernelX));
						float ykern = (Y_kernel.at<float>(kernelY, kernelX));
						xintensity += ((tempInt)* xkern);
						yintensity += ((tempInt)* ykern);
				}
			}
			// Update variables and calculate magnitude and direction of vector
			int xcolor = abs(xintensity / 3);
			int ycolor = abs(yintensity / 3);
			int magnitude = sqrt(xcolor*xcolor + ycolor*ycolor);
			avg_magnitude += magnitude;

			// Calculate the direction/angle of the vector
			if (xintensity != 0) {
				angle = atan((yintensity / 3) / (xintensity / 3));
			}
			else {
				if (yintensity > 0) {
					angle = M_PI / 2;
				}
				if (yintensity < 0) {
					angle = -M_PI / 2;
				}
			}
			avg_direction += angle;
			counter++;

			if ((x_col % interval_width) == 0 && (y_row % interval_height) == 0) {
				std::cout << "counter is " << counter << endl;

				// Find the average direction of all
				avg_direction = avg_direction / counter;
				avg_magnitude = avg_magnitude / counter;

				std::cout << "**************************************" << endl;
				std::cout << "Inside if. Avg dir: " << avg_direction << endl;

				// Find the midpoint of grid and draw lines from there
				Point midpoint, pt1, pt2;
				midpoint.x = (x_col)+interval_width / 2;
				midpoint.y = (y_row)+interval_height / 2;

				/*********************************************************************************/
				// Calculate the points to connect and their slope using the angle
				pt2.x = (int)round(midpoint.x + (interval_width / 2) * cos(avg_direction));
				pt2.y = (int)round(midpoint.y + (interval_height / 2) * sin(avg_direction));

				pt1.x = (int)round(midpoint.x - (interval_width / 2) * cos(avg_direction));
				pt1.y = (int)round(midpoint.y - (interval_height / 2) * sin(avg_direction));

				// Conditions in case the points are off the image grid
				std::cout << "\t\t\t\t\t\t\t midpoint is: " << midpoint << endl;
				std::cout << "\t\t\t\t\t\t\t pt1: " << pt1 << " pt2: " << pt2 << endl;

				if ((pt1.x >= 0 & pt1.x <= src[i].cols) && (pt1.y >= 0 & pt1.y <= src[i].rows)
										&& (pt2.x >= 0 & pt2.x <= src[i].cols ) && (pt2.y >= 0 & pt2.y <= src[i].rows)) {
				// Now we want to draw a vector line
				arrowedLine(tempo, pt1, pt2, Scalar(130, avg_magnitude, 216), 4, CV_AA);

				// Reset parameters
				avg_direction = 0;
				avg_magnitude = 0;
				counter = 0;
				}
			}
			/*// write in color results to the destination file
			* dst.at<uchar>(y_row * step, x_col * step) = 255 - magnitude;
			* Here we can apply more direct constraints such that we truncate certain values.
			* This can be used if we only want to include things that are distinctly in a
			* specific direction (for example make sure x-direction values are > 80 intensity
			*
			//			 if (color >= 50) {
			//			 dst.at<uchar>(y_row * step,x_col * step) = 0;
			//			 }
			//			 if (color < 50) {
			//				 dst.at<uchar>(y_row * step, x_col * step) = 255;
			//			 } */
		}
	}
//	CreateWindow("attempt", tempo);
	//	imwrite( "EditedData.jpg", tempo);
	}
	return tempo;
}


void ImageAnalysis(std::vector<Mat> images, std::vector<Mat> dst, Mat kernel, int kHeight, int kWidth, float delta, int number) {
	int imgHeight = 0, imgWidth = 0;
	std::vector<Mat> final, finalz;
	// Determining downsampling step size
	int step = StepSize(delta);

	for (int i = 0; i < images.size(); i++) {
		imgHeight = images[i].rows;
		imgWidth = images[i].cols;

		// Begin convolution procedure
		dst[i] = FixDimensions(images[i], dst[i], delta);
//		CreateWindow("Destination", dst[i]);
		xyGaussianFilter(images[i], dst[i], kernel, imgHeight, imgWidth, kHeight, kWidth, step);

		// Save Gaussian Blurred Image
		string blurimage = "VertStripes";
		blurimage += std::to_string(number);
		blurimage += ".jpg";
		imwrite(blurimage, dst[i]);

		// Make a copy to save the final image
		final.reserve(dst.size());
		copy(dst.begin(), dst.end(),back_inserter(final));
		final[i] = xyDirectionalFilter(dst[i], final[i], step, false, 15, 15);
		CreateWindow("final", final[i]);

		// Save Vector Data Image
		string text = "VertStripes";
		text += std::to_string(number);
		text += ".jpg";
		imwrite(text, dst[i]);

	}
}


/** @function main */
int main ( int argc, char** argv ) {
  // Declare variables
  vector<Mat> src;
  vector<Mat> dst;
  Mat kernel;
  Point anchor;

  // Initialize arguments for the filter
  float delta = 0.5;
  int kernel_rows = 5, kernel_cols = 5;
  src.reserve(15);
  dst.reserve(15);

  // Check if kernel is appropriate dimensions
  if ((kernel_rows % 2 == 0) && (kernel_cols % 2 == 0)) {
	  // Make an odd kernel
	  kernel_rows+=1;
	  kernel_cols+=1;
  }

  for (int i = 0; i < 1; i++) {
	  // Load an image  CV_LOAD_IMAGE_GRAYSCALE, CV_LOAD_IMAGE_COLOR
	  Mat image = imread("Stripes.png", CV_LOAD_IMAGE_COLOR);
	  src.push_back(image);

	  // Check for invalid input
	  if( !(src[i].data) ) { return -1; }
	  cout << "SIZE OF SRC: " << "\t Width:" << src[i].cols << "\t Height: " << src[i].rows << endl;
  }
  // Copy src array in dst
  dst.reserve(src.size());
  copy(src.begin(), src.end(),back_inserter(dst));

  // Apply filter wanted
  dst[0] = FixDimensions(src[0], dst[0], delta);

//  ImageAnalysis(src, dst, kernel, 7,7, delta,4);
//  ImageAnalysis(src, dst, kernel, kernel_rows, kernel_cols, 0.0, 1);
  ImageAnalysis(src, dst, kernel, kernel_rows, kernel_cols, 0.50, 2);
  ImageAnalysis(src, dst, kernel, kernel_rows, kernel_cols, 0.25, 3);

  return 0;
}

