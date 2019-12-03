/*
 * Description: 	header files for db scan and disjoint partitioning class 
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
/* -------------------------------------------auto kmeans contours -------------------------------------------*/

class Kmeansauto
{
	public:
		vector<Point2f> pdata;
		int distance;
		int nclusters;
		vector<Point2f> centroids;
		vector<int> labels;
		std::map<int, int> grouplabels;

		Kmeansauto(std::vector<cv::Point2f> _pdata, int _distance): pdata(_pdata), distance(_distance){
			// for(auto x: contours){for(auto y: x){pdata.push_back(y);}} //unroll cintours vector and push back in one vector of points
			nclusters = 0;
		}; // second initialize with contours (group of points)
		~Kmeansauto(){
			// std::cout << "Kmeans object deleted" << std::endl; //will run itself
		}; //use brackets otherswise produces an error

		void run(){

			Mat cntlabels, centers; //labels of contours function and the centers returned. This will be used to fuill the centroids

			//partitions based on distance
			int sqrtDistance = distance * distance; // criteria for treshold
			nclusters = cv::partition(pdata, labels, [sqrtDistance](const Point& lhs, const Point& rhs) {
					return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < sqrtDistance;
					});

			//get sorted point groups
			vector<vector<Point2f>> pgroups = getPointGroups();

			centroids = getCentroids(); //also sets grouplabels
			for(int k=0; k<centroids.size();k++){grouplabels[k] = k;} //grouplabels is a map
		}

		std::vector<std::vector<cv::Point2f> > getPointGroups(){
			std::vector<std::vector<cv::Point2f>> pgroups(nclusters);
			for (int i = 0; i < pdata.size(); ++i)
			{
				pgroups[labels[i]].push_back(pdata[i]); //
			}
			return pgroups;
		}

		vector<Point2f> getCentroids(){
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
			int k=0;
			for(int i=0;i<pdata.size();i++)
			{
			     putText(img_color,to_string(labels[i]),pdata[i],  FONT_HERSHEY_COMPLEX,.5,_color,size);
			}
		} 
		void drawCentroids(cv::Mat &img, cv::Scalar color,int size){
			for(auto p: centroids){
        		circle(img, Point2d(floor(p.x),floor(p.y)), size, color, -1);//, 8, 0 );
    		}
		}
		void drawCentroidLabels(cv::Mat &img_color, cv::Scalar _color, int size){
			int k=0;
			for(int i=0;i<centroids.size();i++)
			{
			     putText(img_color,to_string(grouplabels[i]),centroids[i],  FONT_HERSHEY_COMPLEX,.5,_color,size);
			}
		}    
};
/* -------------------------------------------dbscan -------------------------------------------*/

class DbScan
{
	public:
		std::map<int, int> labels;
		std::map<int, int> grouplabels;
		vector<Rect>& data;
		vector<Point2f> centroids;
		int C;
		double eps;
		int mnpts;
		double* dp;
		int nclusters;
		//memoization table in case of complex dist functions
#define DP(i,j) dp[(data.size()*i)+j]
		DbScan(vector<Rect>& _data,double _eps,int _mnpts):data(_data)
	{
		C=-1;
		for(int i=0;i<data.size();i++)
		{
			labels[i]=-99;
		}
		eps=_eps;
		mnpts=_mnpts;
	}
	//destructor
	~DbScan(){
			// std::cout << "Dbscan object deleted" << std::endl; will run itself
	}

		void run()
		{
			dp = new double[data.size()*data.size()];
			for(int i=0;i<data.size();i++)
			{
				for(int j=0;j<data.size();j++)
				{
					if(i==j)
						DP(i,j)=0;
					else
						DP(i,j)=-1;
				}
			}
			for(int i=0;i<data.size();i++)
			{
				if(!isVisited(i))
				{
					vector<int> neighbours = regionQuery(i);
					if(neighbours.size()<mnpts)
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
			for(int k=0; k<centroids.size();k++){grouplabels[k] = k;}
			nclusters = C+1;

		}
		void expandCluster(int p,vector<int> neighbours)
		{
			labels[p]=C;
			for(int i=0;i<neighbours.size();i++)
			{
				if(!isVisited(neighbours[i]))
				{
					labels[neighbours[i]]=C;
					vector<int> neighbours_p = regionQuery(neighbours[i]);
					if (neighbours_p.size() >= mnpts)
					{
						expandCluster(neighbours[i],neighbours_p);
					}
				}
			}
		}

		bool isVisited(int i)
		{
			return labels[i]!=-99;
		}

		vector<int> regionQuery(int p)
		{
			vector<int> res;
			for(int i=0;i<data.size();i++)
			{
				if(distanceFunc(p,i)<=eps)
				{
					res.push_back(i);
				}
			}
			return res;
		}

		double dist2d(Point2d a,Point2d b)
		{
			return sqrt(pow(a.x-b.x,2) + pow(a.y-b.y,2));
		}

		double distanceFunc(int ai,int bi)
		{
			if(DP(ai,bi)!=-1)
				return DP(ai,bi);
			Rect a = data[ai];
			Rect b = data[bi];

			Point2d tla =Point2d(a.x,a.y);
			Point2d tra =Point2d(a.x+a.width,a.y);
			Point2d bla =Point2d(a.x,a.y+a.height);
			Point2d bra =Point2d(a.x+a.width,a.y+a.height);

			Point2d tlb =Point2d(b.x,b.y);
			Point2d trb =Point2d(b.x+b.width,b.y);
			Point2d blb =Point2d(b.x,b.y+b.height);
			Point2d brb =Point2d(b.x+b.width,b.y+b.height);

			double minDist = 9999999;

			minDist = min(minDist,dist2d(tla,tlb));
			minDist = min(minDist,dist2d(tla,trb));
			minDist = min(minDist,dist2d(tla,blb));
			minDist = min(minDist,dist2d(tla,brb));

			minDist = min(minDist,dist2d(tra,tlb));
			minDist = min(minDist,dist2d(tra,trb));
			minDist = min(minDist,dist2d(tra,blb));
			minDist = min(minDist,dist2d(tra,brb));

			minDist = min(minDist,dist2d(bla,tlb));
			minDist = min(minDist,dist2d(bla,trb));
			minDist = min(minDist,dist2d(bla,blb));
			minDist = min(minDist,dist2d(bla,brb));

			minDist = min(minDist,dist2d(bra,tlb));
			minDist = min(minDist,dist2d(bra,trb));
			minDist = min(minDist,dist2d(bra,blb));
			minDist = min(minDist,dist2d(bra,brb));
			DP(ai,bi)=minDist;
			DP(bi,ai)=minDist;
			return DP(ai,bi);
		}

		vector<vector<Rect> > getRectGroups()
		{
			vector<vector<Rect> > ret;
			for(int i=0;i<=C;i++)
			{
				ret.push_back(vector<Rect>());
				for(int j=0;j<data.size();j++)
				{
					if(labels[j]==i)
					{
						// grouplabels[i] = i;
						ret[ret.size()-1].push_back(data[j]);
					}
				}
			}
			return ret;
		}

		vector<vector<Point2f> > getPointGroups()
		{
			vector<vector<Point2f> > prect;
			for(int i=0;i<=C;i++)
			{
				prect.push_back(vector<Point2f>());
				for(int j=0;j<data.size();j++)
				{
					if(labels[j]==i)
					{
						prect[prect.size()-1].push_back((static_cast<cv::Point2f>(data[j].br() + data[j].tl()))*0.5); // one static cast is enough
					}
				}
			}
			return prect;
		}

		vector<Point2f> getCentroids(){
			vector<vector<cv::Point2f>> pgroups = getPointGroups();
			vector <cv::Point2f> mc(pgroups.size());
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

		vector <Point2f> getRectPoints(){
			vector <Point2f> prect(data.size());
			for(int i=0; i< data.size(); i++)
				prect[i] = (static_cast<cv::Point2f>(data[i].br() + data[i].tl()))*0.5;
			// prect[i] = (data[i].br() + data[i].tl())*0.5;
			return prect;
		}
		void drawCentroids(cv::Mat &img, cv::Scalar color,int size){
			for(auto p: centroids){
        		circle(img, Point2d(floor(p.x),floor(p.y)), size, color, -1);//, 8, 0 );
    		}
		}
		void drawDataLabels(cv::Mat &img_color, cv::Scalar _color, int size){
			int k=0;
			for(int i=0;i<data.size();i++)
			{
			    Scalar color;
			    if(labels[i]==-1)
			    {
			        color=Scalar(128,128,128);
			    }else
			    {
			        int label=labels[i];
			        color=_color;
			    }
			     putText(img_color,to_string(labels[i]),data[i].tl(),  FONT_HERSHEY_COMPLEX,.5,color,size);
			}
		} 

		void drawCentroidLabels(cv::Mat &img_color, cv::Scalar _color, int size){
			int k=0;
			for(int i=0;i<centroids.size();i++)
			{
			     putText(img_color,to_string(grouplabels[i]),centroids[i],  FONT_HERSHEY_COMPLEX,.5,_color,size);
			}
		} 
};


/* -------------------------------------------helper funcs -------------------------------------------*/
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
// 