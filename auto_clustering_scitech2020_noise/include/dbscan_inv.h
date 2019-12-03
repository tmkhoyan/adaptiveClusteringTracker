/*
 * Description: 	header files for inverse db scan formulation
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

template <typename T>
inline void printVector__(const std::vector <T> &v){
	int k =0;
	std::for_each(v.begin(),v.end(),
			[&k](const T &s){std::cout << k++ << ": " << s << std::endl;});
}
template <typename T, typename S>
inline void printMap__(const std::map<T,S> m){
	int k =0;
	// for(auto p: m)
	//  std::cout << k++ << ": " <<p.first << ", "<< p.second << std::endl;
	std::for_each(m.begin(),m.end(),
			[&k](std::pair<const T,S> index){std::cout << k++ <<": " <<index.first << ", \t " <<index.second << std::endl;});
}
/* -------------------------------------------dbscan -------------------------------------------*/

class DbScan
{
	public:
		std::map<int, int> labels;
		std::map<int, int> grouplabels;
		std::map<int, int> grouplabels_reordered; // is empty at first reorder then we fill it using last reordered index
		std::vector<unsigned int> lastReorderedIndex; // after reordering such that we keep track of last reorder
		bool isReordered=false;
		std::vector<cv::Rect>& data;
		std::vector<cv::Point2f> centroids;
		int C;
		double eps;
		int mnpts;
		int maxpts; // inverse dbscan parameter 
		double* dp;
		int nclusters;
		//memoization table in case of complex dist functions
#define DP(i,j) dp[(data.size()*i)+j]
		DbScan(std::vector<cv::Rect>& _data,double _eps,int _mnpts):data(_data)
	{
		C=-1;
		for(size_t i=0;i<data.size();i++)
		{
			labels[i]=-99;
		}
		eps=_eps;
		mnpts=_mnpts;
		maxpts = _mnpts+1;
		// maxpts = std::max(_mnpts,_maxpts); // at least as large as maxpts
	}
		//second
		DbScan(std::vector<cv::Rect>& _data,double _eps,int _mnpts,int _maxpts):data(_data)
	{
		C=-1;
		for(size_t i=0;i<data.size();i++)
		{
			labels[i]=-99;
		}
		eps=_eps;
		mnpts=_mnpts;
		// inverse dbscan parameter nescessary constraint
		maxpts = std::max(_mnpts+1,_maxpts); // at least as large as maxpts
	}
		//destructor
		~DbScan(){
			// std::cout << "Dbscan object deleted" << std::endl; will run itself
		}

		void run()
		{
			dp = new double[data.size()*data.size()];

			for(size_t i=0;i<data.size();i++)
			{

				for(size_t j=0;j<data.size();j++)
				{
					if(i==j)
						DP(i,j)=0;
					else
						DP(i,j)=-1;
				}
			}
			for(size_t i=0;i<data.size();i++)
			{
				// std::cout << i << std::endl;
				if(!isVisited(i))
				{
					std::vector<int> neighbours = regionQuery(i);
					// inverse dbscan sorting
					if(neighbours.size()<(size_t)mnpts or neighbours.size()>(size_t)maxpts) // upper treshold 
					{
						labels[i]=-1;//noise
					}else
					{
						C++;
						expandCluster(i,neighbours);
					}
				}
			}
			delete [] dp;
			// get centroids
			centroids = getCentroids(); //remove this and make clas thinner if not requred
			for(size_t k=0; k<centroids.size();k++){grouplabels[k] = k;} // cetroids are already sorted on their id. So grouplabels has equal indeces. sort if required
			// grouplabels_reordered(grouplabels);
			nclusters = C+1;
			// generate index vector
			std::vector<unsigned int> idx(nclusters); // kep int to deal with -1 
			std::iota(idx.begin(), idx.end(), 0); // fill index 
			lastReorderedIndex = idx; // initial index just 1:nclusters
		}

		void expandCluster(int p,std::vector<int> neighbours)
		{
			labels[p]=C;
			for(size_t i=0;i<neighbours.size();i++)
			{
				if(!isVisited(neighbours[i]))
				{
					labels[neighbours[i]]=C;
					std::vector<int> neighbours_p = regionQuery(neighbours[i]);
					if ( (neighbours_p.size() >= (size_t)mnpts) and (neighbours_p.size() < (size_t)maxpts))
					{
						// std::cout << "neighbours_p.size() " << neighbours_p.size() 
						// << ",mnpts " << mnpts 
						// << ",eps "  << eps
						// << ",max "  << maxpts << std::endl;
						expandCluster(neighbours[i],neighbours_p);
					}
				}
			}
		}

		bool isVisited(int i)
		{
			return labels[i]!=-99;
		}

		std::vector<int> regionQuery(int p)
		{
			std::vector<int> res;
			for(size_t i=0;i<data.size();i++)
			{
				if(distanceFunc(p,i)<=eps)
				{
					res.push_back(i);
				}
			}
			return res;
		}

		double dist2d(cv::Point2d a,cv::Point2d b)
		{
			return sqrt(pow(a.x-b.x,2) + pow(a.y-b.y,2));
		}

		double distanceFunc(int ai,int bi)
		{
			if(DP(ai,bi)!=-1)
				return DP(ai,bi);
			cv::Rect a = data[ai];
			cv::Rect b = data[bi];
			/*
			   Point2d cena= Point2d(a.x+a.width/2,
			   a.y+a.height/2);
			   Point2d cenb = Point2d(b.x+b.width/2,
			   b.y+b.height/2);
			   double dist = sqrt(pow(cena.x-cenb.x,2) + pow(cena.y-cenb.y,2));
			   DP(ai,bi)=dist;
			   DP(bi,ai)=dist;*/
			cv::Point2d tla =cv::Point2d(a.x,a.y);
			cv::Point2d tra =cv::Point2d(a.x+a.width,a.y);
			cv::Point2d bla =cv::Point2d(a.x,a.y+a.height);
			cv::Point2d bra =cv::Point2d(a.x+a.width,a.y+a.height);

			cv::Point2d tlb =cv::Point2d(b.x,b.y);
			cv::Point2d trb =cv::Point2d(b.x+b.width,b.y);
			cv::Point2d blb =cv::Point2d(b.x,b.y+b.height);
			cv::Point2d brb =cv::Point2d(b.x+b.width,b.y+b.height);

			double minDist = 9999999;

			minDist = std::min(minDist,dist2d(tla,tlb));
			minDist = std::min(minDist,dist2d(tla,trb));
			minDist = std::min(minDist,dist2d(tla,blb));
			minDist = std::min(minDist,dist2d(tla,brb));

			minDist = std::min(minDist,dist2d(tra,tlb));
			minDist = std::min(minDist,dist2d(tra,trb));
			minDist = std::min(minDist,dist2d(tra,blb));
			minDist = std::min(minDist,dist2d(tra,brb));

			minDist = std::min(minDist,dist2d(bla,tlb));
			minDist = std::min(minDist,dist2d(bla,trb));
			minDist = std::min(minDist,dist2d(bla,blb));
			minDist = std::min(minDist,dist2d(bla,brb));

			minDist = std::min(minDist,dist2d(bra,tlb));
			minDist = std::min(minDist,dist2d(bra,trb));
			minDist = std::min(minDist,dist2d(bra,blb));
			minDist = std::min(minDist,dist2d(bra,brb));
			DP(ai,bi)=minDist;
			DP(bi,ai)=minDist;
			return DP(ai,bi);
		}
		// reoder any vector with index map --> second is the target index order
		template <typename T>
			inline  std::vector<T> reorderVectorMap(std::vector<T> &v,const std::map<int, int> &indexMap){
				std::vector<T> sorted;

				for(auto idx: indexMap){
					sorted.push_back(v[idx.second]);
					// std::cout << "idx first: " << idx.first  << ","  << "idx second: " << idx.second << std::endl;
				}

				return sorted;
			}
		template <typename T>
			inline  std::vector<T> reorderVector(const std::vector<T> &v,const std::vector<unsigned int> &indexVector){
				std::vector<T> sorted;
				std::for_each(indexVector.begin(),indexVector.end(),
						[&v,&sorted](const unsigned int & index){sorted.push_back(v[index]);});
				return sorted;
			}  		
		std::vector<std::vector<cv::Rect> > getRectGroups()
		{
			std::vector<std::vector<cv::Rect> > prect;
			for(int i=0;i<=C;i++)
			{
				prect.push_back(std::vector<cv::Rect>());
				for(size_t j=0;j<data.size();j++)
				{
					if(labels[j]==i)
					{
						prect[prect.size()-1].push_back(data[j]);
					}
				}
			}				

			return (grouplabels_reordered.empty()? prect : reorderVectorMap(prect,grouplabels_reordered));
			// return prect;
		}

		std::vector<std::vector<cv::Point2f> > getPointGroups()
		{
			std::vector<std::vector<cv::Point2f> > prect;
			for(int i=0;i<=C;i++)
			{
				prect.push_back(std::vector<cv::Point2f>());
				for(size_t j=0;j<data.size();j++)
				{
					if(labels[j]==i)
					{
						prect[prect.size()-1].push_back((static_cast<cv::Point2f>(data[j].br() + data[j].tl()))*0.5); // one static cast is enough
					}
				}
			}
			return (grouplabels_reordered.empty()? prect : reorderVectorMap(prect,grouplabels_reordered));
		}

		std::vector<cv::Point2f> getCentroids(){
			//consider 
			std::vector<std::vector<cv::Point2f>> pgroups = getPointGroups(); // sorts and reorders if nescessary
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
				// grouplabels[k] = k; //groups already sorted
				k++;
			}
			return mc;
		}     

		std::vector <cv::Point2f> getRectPoints(){
			std::vector <cv::Point2f> prect(data.size());
			for(size_t i=0; i< data.size(); i++)
				prect[i] = (static_cast<cv::Point2f>(data[i].br() + data[i].tl()))*0.5;
			// prect[i] = (data[i].br() + data[i].tl())*0.5;
			// return prect;
			return (grouplabels_reordered.empty()? prect : reorderVectorMap(prect,grouplabels_reordered));
		}
		// defaiult when no w and h provided take average of ckluster point rectangle sizes!
		std::vector<cv::Rect> getRectCentroids(){

			std::vector<std::vector<cv::Rect> > RectGroup = getRectGroups(); // needed to calc with and height of rect
			std::vector <cv::Rect> centroidRects(centroids.size());

			for(size_t k=0; k<centroids.size();k++){
				float sumw = 0; float sumh =0;
				size_t size_ = RectGroup[k].size();
				for(auto grect: RectGroup[k]){
					//average with and height
					sumw  += grect.width;
					sumh  += grect.height;
				}
				float centrx = centroids[k].x ;
				float centry = centroids[k].y ;
				sumw /= size_;
				sumh /= size_;      
				centroidRects[k] = cv::Rect(centrx - sumw/2,centry- sumh/2,sumw,sumh);
			}
			return centroidRects;
		} 
		//. override and use w h provided!
		std::vector<cv::Rect> getRectCentroids(int w, int h){

			std::vector <cv::Rect> centroidRects(centroids.size());
			for(size_t k=0; k<centroids.size();k++){
				float sumw = 0; float sumh =0;
				float centrx = centroids[k].x ;
				float centry = centroids[k].y ;
				sumw = w;
				sumh = h;      
				centroidRects[k] = cv::Rect(centrx - sumw/2,centry- sumh/2,sumw,sumh);
			}
			return centroidRects;
		}
		//reordercentrooids vector based on idx provided 
		inline void reorderCentroids(const std::vector<unsigned int> &indexVector){
			std::vector <cv::Point2f> centroids_reordered_;
			std::map<int, int> grouplabels_reordered_;

			int k = 0;
			for(auto idx: indexVector){
				centroids_reordered_.push_back(centroids[idx]);
				grouplabels_reordered_[k++] = lastReorderedIndex[idx]; // key(sorted index) --> value (desired index)
				// std::cout << "idx " << idx <<  ",lastReorderedIndex[idx] " << lastReorderedIndex[idx] <<  std::endl;

			}
			grouplabels_reordered = grouplabels_reordered_;

			// reoder index vector so that we cam keep rotating
			lastReorderedIndex = reorderVector(lastReorderedIndex,indexVector);

			centroids = (indexVector.size() == centroids.size())? std::move(centroids_reordered_) : centroids_reordered_;
			isReordered = true;
		}
		/* -------------------------------------------drawing funcs -------------------------------------------*/

		void drawCentroids(cv::Mat &img, cv::Scalar color,int size){
			for(auto p: centroids){
				circle(img, cv::Point2d(floor(p.x),floor(p.y)), size, color, -1);//, 8, 0 );
			}
		}
		void drawDataLabels(cv::Mat &img_color, cv::Scalar _color, int size){
			// int k=0;
			for(size_t i=0;i<data.size();i++)
			{
				cv::Scalar color;
				if(labels[i]==-1)
				{
					color=cv::Scalar(128,128,128);
				}else
				{
					// int label=labels[i];
					color=_color;
				}
				cv::putText(img_color,std::to_string(labels[i]),data[i].tl(),  cv::FONT_HERSHEY_COMPLEX,.5,color,size);
			}
		} 

		void drawCentroidLabels(cv::Mat &img_color, cv::Scalar _color, int size){
			// int k=0;
			for(size_t i=0;i<centroids.size();i++)
			{
				cv::putText(img_color,std::to_string(grouplabels[i]),centroids[i],  cv::FONT_HERSHEY_COMPLEX,.5,_color,size);
			}
		}
		template<typename T>
		void drawCentroidLabels(cv::Mat &img_color, cv::Scalar _color, int size, std::vector<T> fromToMap, cv::Point2f offset=cv::Point2f(15,-15)){
			// int k=0;
			for(size_t i=0;i<centroids.size();i++)
				cv::putText(img_color,std::to_string(grouplabels[i]),centroids[i],  cv::FONT_HERSHEY_COMPLEX,.5,_color,size);

			// if empty will not draw anything 
			for(size_t i=0;i<fromToMap.size();i++)
				cv::putText(img_color,"[" + std::to_string(fromToMap[i]) + "]",centroids[i] +offset,  cv::FONT_HERSHEY_COMPLEX,.5,_color,size);

		}
		// draw the rectangles of all points invariant to order 
		void drawDataBoxes(cv::Mat &img_color, cv::Scalar _color, int size){
			// int k=0;
			cv::RNG rng(12345);
			std::vector<cv::Scalar> colorMap;
			for(int i=0;i<nclusters;i++){
				colorMap.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
			}

			for(size_t i=0;i<data.size();i++)
			{

				cv::rectangle(img_color,data[i], colorMap[labels[i]],size,1);
			}
		} 
		// draw the rectangles of cluster
		void drawClusterBoxes(cv::Mat &img_color, cv::Scalar _color, int size){
			// int k=0;
			cv::RNG rng(12345);
			std::vector<cv::Scalar> colorMap;
			for(int i=0;i<nclusters;i++){
				colorMap.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
			}
			// get cluster center

			for(size_t i=0;i<data.size();i++)
			{

				cv::rectangle(img_color,data[i], colorMap[labels[i]],size,1);
			}
		} 
};

/* -------------------------------------------helper funcs -------------------------------------------*/
// consider renaming to prevent confusion
	template <class T>
inline std::string to_string (const T& t)
{
	std::stringstream ss;
	ss << t;
	return ss.str();
}

std::vector <std::vector<cv::Point>> getContours(cv::Mat img){
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	return contours;
}

std::vector<cv::Rect> getRectPoints(std::vector <std::vector<cv::Point>> contours){
	std::vector<cv::Rect> boxes;
	for(auto pgroups: contours){boxes.push_back(boundingRect(pgroups));}
	return boxes;
}

std::vector<cv::Point2f> getGroupPoints(std::vector <std::vector<cv::Point>> contours){ 
	std::vector<cv::Point2f> points;
	for(auto pgroup: contours){for(auto p: pgroup){points.push_back(p);}} //unroll cintours vector and push back in one vector of points
	return points;
}
