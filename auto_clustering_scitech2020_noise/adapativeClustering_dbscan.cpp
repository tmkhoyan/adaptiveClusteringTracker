/*
 * Description:   autoselection of cluster for automatic determinatio of initial point for object tracking 
 *          example usage:  
 *
 * Author:      Tigran Mkhoyan
 * Email :      t.mkhoyan@tudelft.nl
 */

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <numeric>


#include <unistd.h>
#include <termios.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <exception>

// non-inverse dbscan formulation (original) when uncommented
#define DB_SCAN_MODE_ORIGINAL // comment to use inverse dbscan

using namespace cv;
using namespace std;

#include "./include/hsv_filter.h"

#ifdef DB_SCAN_MODE_ORIGINAL
  #include "./include/clustering.h"
#else
// inverse db scan formukations 
  #include "./include/clustering_mod.h"
  #include "./include/dbscan_mod.h"
#endif

// #define DISTPART  disjoint partitioning

const string filteredwindow = "Filered";

    /* ----------------------------------------------   functions -----------------------------------------------------*/

void gaussNoise(cv::Mat &original_img, double mean_noise=0, double sigma_noise=0.5);
std::string getTime(std::string format="%d_%m_%Y[%H-%M-%S]");



inline void getFps(double &fps, std::time_t timeNow, int &tick, long &frameCounter){
  if (timeNow - tick >= 1){
    tick++;
    fps = frameCounter;
      // cout << "Frames per second: " << frameCounter << 
    frameCounter = 0;
  }
}

inline std::string makeValidDir(const std::string & str){

if (mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
{
    if( errno == EEXIST ) { // alredy exists
      std::cout << "folder: " << str << " --- exists. " << std::endl;
    } else {
        std::cout << "cannot create folder error:" << strerror(errno) << std::endl;
        exit(0);
    }
}
  return (str.back() == '/' ? str : (str +"/"));
}
    /* ----------------------------------------------  main -----------------------------------------------------*/

cv::FileStorage fs_o;
cv::FileStorage fs_img;

VideoWriter video;

int main( int argc, char** argv){

  /* ----------------------------  parse input   -----------------------------------------*/

  if(argc<2){
    cout << "No arguments priovided. Usage cmd [path img file] + " 
       << "{optional}[output path]" 
       << "{optional}[nstep to save images ]" 
       << endl;
       return -1;
  }

  std::string img_input_file = (argc>1)? argv[1] : "";
  std::string im_out_path = (argc>2)? argv[2] : "data_out/";
  int nstep = (argc>3)? atoi(argv[3]) : 1;


// check directories are valid
  makeValidDir(im_out_path);

  std::string eval_noise_path = im_out_path+"/noise_eval/";
  std::string binary_img_path = im_out_path+"/binary/";

  // % removes trailing /?
  makeValidDir(eval_noise_path);
  makeValidDir(binary_img_path);

  
  //put bacjk trainling
  im_out_path = im_out_path+"/";
  /* ----------------------------  initiualise inpout    -----------------------------------------*/

  std::ifstream input(img_input_file);
  fs_o.open(im_out_path+"labels_out.yml", cv::FileStorage::WRITE);
  fs_img.open(im_out_path+"imgs_out.yml", cv::FileStorage::WRITE);

  /* ----------------------------  containers  -----------------------------------------*/
  Mat img_original, img_bw, img_color;
// cv::UMat Umat_in(7343,9456,CV_8UC4);

  std::vector <std::string> lines;
  std::string videosrcName = im_out_path + "cluster_" + getTime() + "_.mkv "; //add trailing white space for gstreamer
  
  /* ----------------------------  store all frames  -----------------------------------------*/
  std::string videosrcName_cluster_noise = im_out_path + "cluster_noise_" + getTime() + "_.mkv "; //add trailing white space for gstreamer
  std::string videosrcName_input= im_out_path + "input_" + getTime() + "_.mkv "; //add trailing white space for gstreamer


  std::string line;

  bool pausePressed = false ;
  int tick = 0;
  double fps=0;
  long frameCounter = 0;
  int recframe_cnt=0;
  int k=0;


  /* ----------------------------  dbscan parameters  -----------------------------------------*/

// // original----
// int dbscan_mnpts = 2;
// int dbscan_epsilon = 20;
// // ---

  int dbscan_mnpts = 1;
  int dbscan_epsilon = 20;
  int dbscan_maxpts = 4;

  //initiate window
  cv::namedWindow(filteredwindow,cv::WINDOW_NORMAL);
  //timing   //timing
  std::time_t timeBegin = std::time(0);



 // Read lines as long as the file is
  while (std::getline(input, line))              
    lines.push_back(line);

  int maxFrames = lines.size();

  std::vector<std::vector<cv::Point2f>> DBscanCentroids; DBscanCentroids.reserve(maxFrames);
  //vector to store images
  std::vector <cv::Mat > outFrames; outFrames.reserve(maxFrames);
  std::vector <cv::Mat > outFrames_hvsth; outFrames_hvsth.reserve(maxFrames);


  /* ----------------------------  load frames into vector first -----------------------------------------*/

  // for(auto imagestr: lines){

  // k = 0;
  // for(auto &s: lines){
  //   outFrames.push_back(cv::imread(s.c_str(),cv::IMREAD_COLOR));
  //   cout << "loading image :"<<k++ << endl;
  // }
  /* ----------------------------------------------  initiate hsv filter -----------------------------------------------------*/

     img_original = cv::imread(lines[0].c_str(),cv::IMREAD_COLOR);
      // img_original = 

  enum HSVwindow window = HSVwindow::ALL;
      Hsvfilter hsvfilter(img_original,4,2); //set erode and dilate elements

      hsvfilter.TUNE = true;
      hsvfilter.init();

      img_color    = cv::Mat::zeros(img_original.size(),CV_8UC3);

        /* ----------------------------------------------  main loop  -----------------------------------------------------*/

      auto keyPressed = -1;

int imageCounter=0;
/* ----------------------------------------------  from disk -----------------------------------------------------*/
    for(auto imagestr: lines){
      Mat img_original = imread(imagestr.c_str(),cv::IMREAD_COLOR);
 
 /* ---------------------------------------------- add noise here ! -----------------------------------------------------*/

          gaussNoise(img_original,0.0,0.4);

          hsvfilter.run(img_original);
      cv::cvtColor(hsvfilter.imgBW,img_color,cv::COLOR_GRAY2RGB);


      cout << "cntr --->>>: " << endl;

			vector<vector<Point>> contours =  getContours(hsvfilter.imgBW); //get contours of the image;
			//containers for data to hold 
			vector <Rect> boxes  = getRectPoints(contours);      // gets the rectangle bounded by contour points
			vector <Point2f> points  = getGroupPoints(contours); //unrolls all points

			//run dbscan
      #ifdef DB_SCAN_MODE_ORIGINAL
			DbScan dbscan(boxes,dbscan_epsilon,dbscan_mnpts); //provde boxes around contours ans distamce plus sigma
		  #else 
      DbScan dbscan(boxes,dbscan_epsilon,dbscan_mnpts,dbscan_maxpts); //provde boxes around contours ans distamce plus sigma
      #endif
      cout << "cb --->>>: " << endl;

      dbscan.run();
			std::vector <cv::Point2f> mc_dbscan = dbscan.centroids;
      cout << "out --->>>: " << endl;

			//get dbscan groups and centroids
			std::vector <std::vector<cv::Rect>> groupedRect = dbscan.getRectGroups();
			std::vector <std::vector<cv::Point2f>> groupedPoint_dbscan = dbscan.getPointGroups();
			std::vector <cv::Point2f> rectPoints = dbscan.getRectPoints();

      // for(auto p: mc_dbscan)
      cout << "points: " << mc_dbscan << endl;
      DBscanCentroids.push_back(mc_dbscan);

      cv::Mat tmp = img_color.clone(); //for drawing noise labels


			//draw
      dbscan.drawCentroids(img_color,CV_RGB(255,0,255),2);   
      dbscan.drawCentroidLabels(img_color,CV_RGB(255,0,255),1);
      // draw data labels including noise
      dbscan.drawDataLabels(tmp, CV_RGB(255,0,255),1);

 /* ----------------------------------------------  show noise tests---------------------------------------------- */

      // cv::Mat combined;  hconcat(img_original,img_color,combined);
      cv::Mat combined;  hconcat(img_original,img_color,combined);
          cv::imshow("noised",combined);

        std::string cnt_str = std::to_string(imageCounter);

        if(!(imageCounter % nstep)){
        cv::imwrite(eval_noise_path + "id_" + cnt_str + "_original_mean=0,0,sig=0,5" +".jpg",  img_original);
        cv::imwrite(eval_noise_path + "id_" + cnt_str + "_hsv_mean=0,0,sig=0,5" +".jpg",       hsvfilter.imageHSV );
        cv::imwrite(eval_noise_path + "id_" + cnt_str + "_thresh_mean=0,0,sig=0,5" +".jpg",    hsvfilter.imgBW );
        cv::imwrite(eval_noise_path + "id_" + cnt_str + "_clustered_mean=0,0,sig=0,5" +".jpg",    img_color );
        cv::imwrite(eval_noise_path + "id_" + cnt_str + "_clustered_noises_mean=0,0,sig=0,5" +".jpg",    tmp );

        cout << "written counter: " << cnt_str << endl;

      }

    imageCounter++;
 /* ----------------------------------------------   get fps ---------------------------------------------- */
    frameCounter++; 
  
    std::time_t timeNow = std::time(0) - timeBegin; // calculates only integer seconds
    getFps(fps,timeNow,tick,frameCounter); 

 /* ---------------------------------------------- put text ---------------------------------------------- */
    #ifdef DB_SCAN_MODE_ORIGINAL
    putText(img_color,"dbscan clusters:     " +to_string(dbscan.nclusters)
      +" [eplsilon=" +to_string(dbscan_epsilon)+", minPts=" +to_string(dbscan_mnpts)+ "]" , Point(100,50), FONT_HERSHEY_COMPLEX,.5,CV_RGB(255,0,255),1);
    #else 
    putText(img_color,"dbscan clusters:     " +to_string(dbscan.nclusters)
      +" [eplsilon=" +to_string(dbscan_epsilon)+", minPts=" +to_string(dbscan_mnpts)
      +", maxPts="  +std::to_string(dbscan.maxpts) + "]" , Point(100,50), FONT_HERSHEY_COMPLEX,.5,CV_RGB(255,0,255),1);
    #endif

    putText(img_color,"fps: "+to_string(fps),Point(600,50),  FONT_HERSHEY_COMPLEX,.5,CV_RGB(0,255,0),1);
  

   /* ---------------------------------------------- showimage ---------------------------------------------- */
      //show and store
    imshow(filteredwindow,img_color);
    cout <<"Frame: " << endl;
    // outFrames[recframe_cnt] = img_color;

    cv::Mat img_color_; img_color.copyTo(img_color_);
    outFrames.push_back(img_color_);
    cv::Mat img_tmp; hsvfilter.imgBW.copyTo(img_tmp);
    outFrames_hvsth.push_back(img_tmp);


      // cout <<"Frame: " <<recframe_cnt << ", fps :" << fps <<endl;
    recframe_cnt = (recframe_cnt>=(maxFrames-1))? 0: recframe_cnt+1;

  // ---------------------------------------------- 
      if(keyPressed=='p' | keyPressed=='P'){ //P or p
        pausePressed = pausePressed==true ? false: true;
      } else if(keyPressed==13){
        break;
      } //enter

      if(pausePressed){
        while(cv::waitKey(50) !=' '){}//waiting for space input to move to next frame
      }

    keyPressed = (cv::waitKey(1));
   // ---------------------------------------------- 

    cout << fps << endl;


  }
    /* ----------------------------------------------  end main image loop  -----------------------------------------------------*/

// writing to yaml doesnt work well, write to jpeg
k = 0; //cout << "writing image data: "<< endl;
for(auto img : outFrames_hvsth){
  // write images vector
  if(!(k % nstep)){
    // fs_img  << "hsv_th" + std::to_string(k) << img;//cv::Mat(img.size(),CV_8U,img.data);;
    cv::imwrite(binary_img_path +  "id_1" + std::to_string(k) +".jpg",img);
    cv::imshow("zz",img);
    cv::waitKey(10);
    cout << "writing image: " << k << endl;
  }

  k++;
}

cout << "written imgages data. "<< endl;

// }

    /* ----------------------------------------------  end main loop write image  -----------------------------------------------------*/

  //save video

// // video.open("appsrc ! autovideoconvert ! x264enc pass=quant ! matroskamux ! filesink location=test_thin.mkv ", 0,(double)20, cv::Size(1088,600), true);
video.open(videosrcName, 0,(double)20, cv::Size(1088,600), true);

std::cout << "nfranmes: " << outFrames.size();
std::cout<< " writing images..." << std::endl;
for(auto frame : outFrames){
  if(!frame.empty()){
    imshow("aa", frame);
    waitKey(1);
    video.write(frame);
  }
}

video.open(videosrcName_cluster_noise, 0,(double)20, cv::Size(1088,600), true);
video.open(videosrcName_input, 0,(double)20, cv::Size(1088,600), true);


for(auto frame : outFrames){
  if(!frame.empty()){
    imshow("aa", frame);
    waitKey(1);
    video.write(frame);
  }
}

for(auto frame : outFrames){
  if(!frame.empty()){
    imshow("aa", frame);
    waitKey(1);
    video.write(frame);
  }
}

cout << "enter to exit" << endl;
while(cv::waitKey(10)!=27){

}

return 0;
}
    /* ----------------------------------------------  end main   -----------------------------------------------------*/


void gaussNoise(cv::Mat &original_img, double mean_noise, double sigma_noise){
 // cv::cvtColor(original_img, noise_img, cv::COLOR_BGR2GRAY);
        
 cv::Mat noise_img = original_img.clone(); 

 cv::RNG rng(cv::getCPUTickCount());
 rng.fill(noise_img,cv::RNG::NORMAL,mean_noise,sigma_noise); // mean and variance 
 noise_img = noise_img*255;

 // noise_img += original_img;
 original_img += noise_img;

}

std::string getTime(std::string format){
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm,format.c_str());
  auto str = oss.str();
  return str;
}

