/*
 * Description:     hsv filter class to apply hsv filtering based on treshold level
 *
 * Author:          Tigran Mkhoyan
 * Email :          t.mkhoyan@tudelft.nl
 */

#include <fstream>
#include <iostream>


extern const string STR_HSVCONF = "hsvconf.txt";

const string originalwindow = "Original Image";
const string HSVwindow = "HSV Image";
const string BWwindow = "Thresholded Image";

typedef cv::Point3_<uint8_t> Pixel;

enum class HSVwindow { IMG, HSV, BW, ALL };

class Hsvfilter
{
	public:
		Mat img;
		Mat imgBW;
		Mat imageHSV;
		bool enableShow = true;
		bool startImage = true;
		bool TUNE = false;
		int img_W = 400;
		int img_H = 600;
		std::fstream hsvconf;
		int H_MIN = 0;
		int H_MAX = 256;
		int S_MIN = 0;
		int S_MAX = 256;
		int V_MIN = 0;
		int V_MAX = 256;
		int pixErode=2;
		int pixDilate=2;
		std::string trackbarWindowName = "Trackbars";

		Hsvfilter(cv::Mat _img): img(_img), imgBW(cv::Mat::zeros(_img.size(),CV_8UC1)){
		};

		Hsvfilter(cv::Mat _img, int _pixErode, int _pixDilate): img(_img), imgBW(cv::Mat::zeros(_img.size(),CV_8UC1)), pixErode(_pixErode), pixDilate(_pixDilate){
		};
		~Hsvfilter(){}; //use brackets otherswise produces an error

		void run(cv::Mat &img_in){
			tresholdImage(img_in);
			filterImage(imgBW);

		}

		void init(){
			if(TUNE) //if manual tuning enabled
				createTrackbars();
   			Mat imageHSV;
    		cv::cvtColor(img, imageHSV, cv::COLOR_BGR2HSV);
    		cv::inRange(imageHSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),imgBW); // get bw image after hsv
			img_H = ((float)img_W/img.size().width)*img.size().height;
			//create named windows
			if(enableShow){
			cv::namedWindow(originalwindow,cv::WINDOW_NORMAL);
			cv::namedWindow(HSVwindow,cv::WINDOW_NORMAL);
			cv::namedWindow(BWwindow,cv::WINDOW_NORMAL);
			}
		}

		static void on_trackbar( int val, void*){
			std::cout << "HSV parameter : " << val << std::endl;
		}

		void createTrackbars(){
			//create window for trackbars
			namedWindow(trackbarWindowName,0);
			std::string line;

			int inH_MIN = 0;
			int inH_MAX = 256;
			int inS_MIN = 0;
			int inS_MAX = 256;
			int inV_MIN = 0;
			int inV_MAX = 256;

			std::ifstream in_hsvconf("hsvconf.txt");

			if(in_hsvconf.is_open()){

				do {
					std::getline( in_hsvconf, line);
					istringstream ss(line);
					ss  >> inH_MIN 
						>> inS_MIN 
						>> inV_MIN 
						>> inH_MAX 
						>> inS_MAX 
						>> inV_MAX;
				std::cout << "HSV settings are: HSV_min/HSV_max " << line << std::endl;
				} while(!line.empty());
				in_hsvconf.close();
			} 
			else {
				std::cout << "hsvconf not present" << std::endl; 
			} 
			//create trackbars with min max vavlues)
			cv::createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
			cv::createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
			cv::createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
			cv::createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
			cv::createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
			cv::createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);

			//set trackbar position
			cv::setTrackbarPos( "H_MIN", trackbarWindowName, inH_MIN);
			cv::setTrackbarPos( "S_MIN", trackbarWindowName, inS_MIN);
			cv::setTrackbarPos( "V_MIN", trackbarWindowName, inV_MIN);
			cv::setTrackbarPos( "H_MAX", trackbarWindowName, inH_MAX);
			cv::setTrackbarPos( "S_MAX", trackbarWindowName, inS_MAX);
			cv::setTrackbarPos( "V_MAX", trackbarWindowName, inV_MAX);

			//set values to the ones read from config
			H_MIN = inH_MIN;
			H_MAX = inH_MAX;
			S_MIN = inS_MIN;
			S_MAX = inS_MAX;
			V_MIN = inV_MIN;
			V_MAX = inV_MAX;
		}

		void tresholdImage(Mat img_in){
			img = img_in;
    		cv::cvtColor(img_in, imageHSV, cv::COLOR_BGR2HSV);
    		cv::inRange(imageHSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),imgBW); // get bw image after hsv
			img_H = ((float)img_W/img.size().width)*img.size().height;
		}

		void filterImage(Mat &thresh){
			//apply erode and dilate to get the visible markers out
			cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE,Size(pixErode,pixErode));
			cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_ELLIPSE,Size(pixDilate,pixDilate));
#ifdef HSV_MORPH_OPS
			cv::erode(thresh,thresh,erodeElement);
			cv::dilate(thresh,thresh,dilateElement);
#endif

		}

		void show(enum HSVwindow window){
			if(enableShow){
			switch(window)
			{
			    case HSVwindow::HSV: {
			    	// Mat imageHSV; 
    				// cv::cvtColor(img, imageHSV, cv::COLOR_BGR2HSV);
					if(startImage)  
    					cv::resizeWindow(HSVwindow,img_W, img_H);
			    	break;
					cv::imshow(HSVwindow,imageHSV);
			    }
			    case HSVwindow::BW:{ 
					cv::imshow(BWwindow,imgBW); 
    				cv::resizeWindow(BWwindow,img_W, img_H);
			    	break;
			    }
			    case HSVwindow::IMG: { 
					if(startImage)
    					cv::resizeWindow(originalwindow,img_W, img_H);
					cv::imshow(originalwindow,img);  
			    	break; 
			    }
			    case HSVwindow::ALL:{ 
			    	// cv::Mat imageHSV; 
    				// cv::cvtColor(img, imageHSV, cv::COLOR_BGR2HSV);
					cv::imshow(HSVwindow,imageHSV);  
					cv::imshow(originalwindow,img);
					cv::imshow(BWwindow,imgBW);
					if(startImage){
    				cv::resizeWindow(originalwindow,img_W, img_H);
    				cv::resizeWindow(HSVwindow,img_W, img_H);
    				cv::resizeWindow(BWwindow,img_W, img_H);
					//move windows
    				cv::moveWindow(HSVwindow,img_W,0);
    				cv::moveWindow(BWwindow,img_W*2,0);
    				}
			    	break;
			    	} 
			}
			startImage = false;
		}
		}

		void setParamHSV(
		int _H_MIN,
		int _H_MAX,
		int _S_MIN,
		int _S_MAX,
		int _V_MIN,
		int _V_MAX){

		H_MIN = _H_MIN;
		H_MAX = _H_MAX;
		S_MIN = _S_MIN;
		S_MAX = _S_MAX;
		V_MIN = _V_MIN;
		V_MAX = _V_MAX;

		}
};

