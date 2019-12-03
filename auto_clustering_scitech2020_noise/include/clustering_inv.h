/*
 * Description: 	header files for disjoint partitioning class 
 * 					compile example -I/usr/local/include/opencv4 
 *            											   -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_imgcodecs
 * 					example usage: 	
 *
 * Author: 			Tigran Mkhoyan
 * Email : 			t.mkhoyan@tudelft.nl
 */

#pragma once

#include <map>
#include <sstream>
#include <math.h>
/* -------------------------------------------auto kmeans contours -------------------------------------------*/

class Kmeansauto
{
	public:

		std::vector<cv::Point2f> pdata;
		int distance;
		int nclusters;
		std::vector<cv::Point2f> centroids;
		std::vector<int> labels;
		std::map<int, int> grouplabels;

		Kmeansauto(std::vector<cv::Point2f> _pdata, int _distance): pdata(_pdata), distance(_distance){
			nclusters = 0;
		}; // second initialize with contours (group of points)
		~Kmeansauto(){
			// std::cout << "Kmeans object deleted" << std::endl; //will run itself
		}; //use brackets otherswise produces an error

		void run(){

			cv::Mat cntlabels, centers; //labels of contours function and the centers returned. This will be used to fuill the centroids

			//partitions based on distance
			int sqrtDistance = distance * distance; // criteria for treshold
			nclusters = cv::partition(pdata, labels, [sqrtDistance](const cv::Point& lhs, const cv::Point& rhs) {
					return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < sqrtDistance;
					});

			//get sorted point groups
			std::vector<std::vector<cv::Point2f>> pgroups = getPointGroups();

			centroids = getCentroids(); //also sets grouplabels
			for(size_t k=0; k<centroids.size();k++){grouplabels[k] = k;} //grouplabels is a map

		}

		std::vector<std::vector<cv::Point2f> > getPointGroups(){
			// You can save all points in the same class in a vector (one for each class), just like findContours
			std::vector<std::vector<cv::Point2f>> pgroups(nclusters);
			// std::vector<std::vector<cv::Point2f>> grouplabels(nclusters);
			for (size_t i = 0; i < pdata.size(); ++i)
			{
				pgroups[labels[i]].push_back(pdata[i]); //
			}
			return pgroups;
		}

		std::vector<cv::Point2f> getCentroids(){
			std::vector<std::vector<cv::Point2f>> pgroups = getPointGroups();
			std::vector <cv::Point2f> mc(pgroups.size());
			int k = 0;
			for(auto pgroup: pgroups){
				float sumx = 0; float sumy =0;
				size_t size = pgroup.size();
				for(auto p: pgroup){
					sumx += p.x;
					sumy += p.y;
				}
				mc[k].x = sumx/size;
				mc[k].y = sumy/size;
				k++;
			}
			return mc;
		}  
		void drawDataLabels(cv::Mat &img_color, cv::Scalar _color, int size){
			// int k=0;
			for(size_t i=0;i<pdata.size();i++)
			{
			     putText(img_color,std::to_string(labels[i]),pdata[i],  cv::FONT_HERSHEY_COMPLEX,.5,_color,size);
			}
		} 
		// invariant to order
		void drawCentroids(cv::Mat &img, cv::Scalar color,int size){
			for(auto p: centroids){
        		cv::circle(img, cv::Point2d(floor(p.x),floor(p.y)), size, color, -1);//, 8, 0 );
    		}
		}
		void drawCentroidLabels(cv::Mat &img_color, cv::Scalar _color, int size){
			// int k=0;
			for(size_t i=0;i<centroids.size();i++)
			{
			     cv::putText(img_color,std::to_string(grouplabels[i]),centroids[i],  cv::FONT_HERSHEY_COMPLEX,.5,_color,size);
			}
		}    
};